#include "common/archive.h"
#include "common/file.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <zlib.h>

namespace {
using Bytes = std::vector<uint8_t>;
void Check(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "ImageArchiveTests: %s\n", what);
    std::abort();
  }
}
void Put(Bytes &bytes, size_t p, uint64_t n, size_t width) {
  Check(p <= bytes.size() && width <= bytes.size() - p,
        "fixture write in bounds");
  for (size_t i = 0; i < width; ++i)
    bytes[p + i] = n >> (i * 8);
}
void Text(Bytes &bytes, size_t p, const std::string &text) {
  Check(p <= bytes.size() && text.size() <= bytes.size() - p,
        "fixture string in bounds");
  std::copy(text.begin(), text.end(), bytes.begin() + p);
}
constexpr size_t Unit = 4096;
constexpr size_t Heap = Unit * 2;
Bytes Payload(size_t size) {
  Bytes result(size);
  std::mt19937 random(47);
  for (size_t i = 0; i < size; ++i)
    result[i] = i < Unit ? 0x5a : random() & 255;
  return result;
}
void Entry(Bytes &image, size_t p, const std::u16string &name, uint32_t first,
           uint64_t size, uint64_t valid, bool directory = false,
           bool contiguous = true) {
  const auto secondary = 1 + (name.size() + 14) / 15;
  image[p] = 0x85;
  image[p + 1] = secondary;
  Put(image, p + 4, directory ? 0x10 : 0x20, 2);
  image[p + 32] = 0xc0;
  image[p + 33] = contiguous ? 3 : 1;
  image[p + 35] = name.size();
  Put(image, p + 40, valid, 8);
  Put(image, p + 52, first, 4);
  Put(image, p + 56, size, 8);
  for (size_t i = 0; i < name.size(); ++i) {
    const auto q = p + 64 + i / 15 * 32;
    image[q] = 0xc1;
    Put(image, q + 2 + i % 15 * 2, name[i], 2);
  }
  uint16_t checksum = 0;
  for (size_t i = 0; i < (secondary + 1) * 32; ++i)
    if (i != 2 && i != 3)
      checksum = uint16_t(((checksum << 15) | (checksum >> 1)) + image[p + i]);
  Put(image, p + 2, checksum, 2);
}
Bytes Exfat() {
  Bytes image(Unit * 40);
  Text(image, 3, "EXFAT   ");
  image[510] = 0x55;
  image[511] = 0xaa;
  Put(image, 72, image.size() / 512, 8);
  Put(image, 80, Unit / 512, 4);
  Put(image, 84, Unit / 512, 4);
  Put(image, 88, Heap / 512, 4);
  Put(image, 92, 38, 4);
  Put(image, 96, 2, 4);
  image[108] = 9;
  image[109] = 3;
  image[110] = 1;
  Put(image, Unit + 2 * 4, 0xffffffff, 4);
  Entry(image, Heap, u"eboot.bin", 3, Unit * 3 + 7, Unit * 3 + 7);
  const auto payload = Payload(Unit * 3 + 7);
  std::copy(payload.begin(), payload.end(), image.begin() + Heap + Unit);
  Entry(image, Heap + 96, u"Empty", 7, Unit, Unit, true);
  Entry(image, Heap + 192, u"zero.bin", 8, 100, 3);
  Text(image, Heap + 6 * Unit, "abcTHIS MUST BE ZEROED");
  Entry(image, Heap + 288, u"rally\U0001f697.bin", 9, Unit + 10, Unit + 10,
        false, false);
  Put(image, Unit + 9 * 4, 11, 4);
  Put(image, Unit + 11 * 4, 0xffffffff, 4);
  std::fill_n(image.begin() + Heap + 7 * Unit, Unit, 0x11);
  std::fill_n(image.begin() + Heap + 9 * Unit, Unit, 0x22);
  return image;
}
Bytes Pfsc(const Bytes &image) {
  const auto count = (image.size() + Unit - 1) / Unit;
  Bytes result(48 + (count + 1) * 8);
  Text(result, 0, "PFSC");
  Put(result, 12, Unit, 4);
  Put(result, 16, Unit, 8);
  Put(result, 24, 48, 8);
  Put(result, 32, result.size(), 8);
  Put(result, 40, image.size(), 8);
  for (size_t i = 0; i < count; ++i) {
    Put(result, 48 + i * 8, result.size(), 8);
    Bytes block(Unit);
    const auto take = std::min(Unit, image.size() - i * Unit);
    std::copy_n(image.begin() + i * Unit, take, block.begin());
    Bytes packed(compressBound(Unit));
    uLongf length = packed.size();
    Check(compress2(packed.data(), &length, block.data(), block.size(), 6) ==
              Z_OK,
          "compress fixture");
    if (length < Unit)
      result.insert(result.end(), packed.begin(), packed.begin() + length);
    else
      result.insert(result.end(), block.begin(), block.end());
  }
  Put(result, 48 + count * 8, result.size(), 8);
  return result;
}
Bytes Pfs(const Bytes &pfsc, unsigned mode = 8, bool fragmented = false,
          bool uroot = false) {
  const auto count = (pfsc.size() + Unit - 1) / Unit;
  const bool signedNodes = (mode & 1) != 0, wide = (mode & 2) != 0;
  const size_t stride = signedNodes ? (wide ? 0x310 : 0x2c8) : 0xa8;
  const size_t width = wide ? 8 : 4, step = width + (signedNodes ? 32 : 0);
  const size_t ptr = (wide ? 0x68 : 0x64) + (signedNodes ? 32 : 0);
  const size_t dataBlock = uroot ? 4 : 3;
  Bytes image((dataBlock + count) * Unit);
  Put(image, 0, 2, 8);
  Put(image, 8, 20130315, 8);
  Put(image, 0x1c, mode, 2);
  Put(image, 0x20, Unit, 4);
  Put(image, 0x30, uroot ? 3 : 2, 8);
  Put(image, 0x40, 1, 8);
  auto directory = [&](size_t inode, size_t block, size_t child, bool root) {
    const auto node = Unit + inode * stride;
    Put(image, node, 0x4000, 2);
    Put(image, node + 8, 32, 8);
    Put(image, node + 0x60, 1, width);
    Put(image, node + ptr, block, width);
    const auto p = Unit * block;
    Put(image, p, child, 4);
    Put(image, p + 4, root ? 3 : 2, 4);
    Put(image, p + 8, root ? 5 : 10, 4);
    Put(image, p + 12, 32, 4);
    Text(image, p + 16, root ? "uroot" : "test.exfat");
  };
  directory(0, 2, 1, uroot);
  if (uroot)
    directory(1, 3, 2, false);
  const auto inode = Unit + (uroot ? 2 : 1) * stride;
  Put(image, inode, 0x8000, 2);
  Put(image, inode + 4, 1, 4);
  Put(image, inode + 8, pfsc.size(), 8);
  Put(image, inode + 16, Unit * 40, 8);
  Put(image, inode + 0x60, count, width);
  if (!fragmented) {
    Put(image, inode + ptr, dataBlock, width);
    Put(image, inode + ptr + step, wide ? UINT64_MAX : UINT32_MAX, width);
    std::copy(pfsc.begin(), pfsc.end(), image.begin() + Unit * dataBlock);
  } else {
    Check(count <= 12, "fragmented fixture fits direct PFS pointers");
    for (size_t i = 0; i < count; ++i) {
      const auto block = dataBlock + count - i - 1;
      Put(image, inode + ptr + i * step, block, width);
      const auto take = std::min(Unit, pfsc.size() - i * Unit);
      std::copy_n(pfsc.begin() + i * Unit, take, image.begin() + block * Unit);
    }
  }
  return image;
}
void Save(const std::filesystem::path &path, const Bytes &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  Check(file.good(), "write image fixture");
}
Bytes Read(Common::File &file, size_t offset, size_t length) {
  Check(file.Seek(offset), "seek");
  Bytes bytes(length);
  uint32_t read = 0;
  file.Read(bytes.data(), length, &read);
  bytes.resize(read);
  return bytes;
}
void Synthetic(const std::filesystem::path &dir) {
  const auto path = dir / "test.FFPFSC";
  const auto exfat = Exfat(), packed = Pfsc(exfat), valid = Pfs(packed);
  Save(path, valid);
  Check(Common::IsSupportedArchive(path), "case-insensitive extension");
  auto reader = Common::OpenArchive(path);
  Check(bool(reader), "open PFS/PFSC/exFAT image");
  const auto root = Common::MakeArchivePath(path);
  Check(Common::File::GetDirEntries(root).size() == 4, "list root");
  Check(Common::File::IsDirectoryExisting(root / "Empty"),
        "keep empty directory");
  Common::File file(root / "EBOOT.BIN", Common::File::Mode::Read);
  Check(!file.IsInvalid(), "case-insensitive member lookup");
  const auto payload = Payload(Unit * 3 + 7);
  Check(Read(file, 0, payload.size()) == payload,
        "compressed and raw blocks match");
  Check(Read(file, Unit - 3, Unit + 23) ==
            Bytes(payload.begin() + Unit - 3, payload.begin() + Unit * 2 + 20),
        "cross block read");
  Check(Read(file, payload.size() - 2, 100).size() == 2, "EOF short read");
  Check(Read(file, UINT64_MAX, 2).empty(), "seek beyond EOF");
  Common::File second(root / "eboot.bin", Common::File::Mode::Read);
  Check(Read(second, 8, 19) == Bytes(payload.begin() + 8, payload.begin() + 27),
        "independent handle");
  Common::File zero(root / "zero.bin", Common::File::Mode::Read);
  auto expected = Bytes(100);
  expected[0] = 'a';
  expected[1] = 'b';
  expected[2] = 'c';
  Check(Read(zero, 0, 100) == expected, "zero uninitialized data tail");
  Common::File unicode(root /
                           std::filesystem::path("rally\xf0\x9f\x9a\x97.bin"),
                       Common::File::Mode::Read);
  expected.assign(Unit + 10, 0x11);
  std::fill(expected.begin() + Unit, expected.end(), 0x22);
  Check(Read(unicode, 0, expected.size()) == expected,
        "UTF16 surrogate and fragmented chain");
  Common::File write(root / "eboot.bin", Common::File::Mode::Write);
  Check(write.IsInvalid(), "reject writes");
  Check(!Common::GetArchiveInfo(root / ".." / "eboot.bin"), "reject traversal");

  for (unsigned mode : {8u, 9u, 11u}) {
    for (bool fragmented : {false, true}) {
      const auto variant =
          dir / ("mode" + std::to_string(mode) +
                 (fragmented ? "-fragmented" : "") + ".ffpfsc");
      Save(variant, Pfs(packed, mode, fragmented, true));
      Common::File variantFile(Common::MakeArchivePath(variant, "eboot.bin"),
                               Common::File::Mode::Read);
      Check(!variantFile.IsInvalid() &&
                Read(variantFile, 0, payload.size()) == payload,
            "signed/wide/fragmented PFS and uroot reads");
    }
  }
  const auto topLevel = dir / "top-level.ffpfsc";
  Save(topLevel, Pfsc(Pfsc(Pfsc(valid))));
  Common::File nested(Common::MakeArchivePath(topLevel, "eboot.bin"),
                      Common::File::Mode::Read);
  Check(!nested.IsInvalid() && Read(nested, 0, payload.size()) == payload,
        "nested PFSC layers share cache");

  auto corrupt = packed;
  uint64_t assetBlock = 0;
  for (size_t i = 0; i < 8; ++i)
    assetBlock |= uint64_t(corrupt[48 + 3 * 8 + i]) << (i * 8);
  corrupt[assetBlock + 5] ^= 0x80;
  const auto damagedPath = dir / "damaged-asset.ffpfsc";
  Save(damagedPath, Pfs(corrupt));
  Common::File damaged(Common::MakeArchivePath(damagedPath, "eboot.bin"),
                       Common::File::Mode::Read);
  Check(!damaged.IsInvalid(),
        "corrupt asset does not prevent opening valid metadata");
  Bytes damagedBytes(payload.size());
  uint32_t damagedCount = 99;
  Check(
      !damaged.Read(damagedBytes.data(), damagedBytes.size(), &damagedCount) &&
          damagedCount == 0 && damaged.Tell() == 0,
      "corrupt asset reports failure and preserves cursor");
  uint32_t eofCount = 99;
  Check(file.Seek(file.Size()), "seek to EOF");
  Check(file.Read(damagedBytes.data(), 1, &eofCount) && eofCount == 0,
        "EOF succeeds and differs from corruption");

  auto reject = [&](Bytes bytes, const char *reason) {
    static size_t serial = 0;
    const auto bad = dir / ("bad" + std::to_string(++serial) + ".ffpfsc");
    Save(bad, bytes);
    Check(!Common::OpenArchive(bad), reason);
  };
  auto broken = valid;
  broken.resize(30);
  reject(broken, "reject truncated header");
  broken = valid;
  Put(broken, 0x1c, 4, 2);
  reject(broken, "reject encryption");
  broken = valid;
  Put(broken, 0x20, 0, 4);
  reject(broken, "reject zero PFS unit");
  broken = valid;
  Put(broken, 0x30, UINT64_MAX, 8);
  reject(broken, "reject excessive inodes");
  broken = packed;
  Put(broken, 48, 0, 8);
  reject(Pfs(broken), "reject overlapping PFSC offsets");
  broken = packed;
  Put(broken, 12, 1, 4);
  reject(Pfs(broken), "reject invalid PFSC unit");
  // Block zero contains the exFAT boot sector and is always read when mounting.
  broken = packed;
  const auto data = 48 + ((exfat.size() + Unit - 1) / Unit + 1) * 8;
  broken[data + 5] ^= 0x80;
  reject(Pfs(broken), "reject corrupt zlib metadata");
  broken = exfat;
  Put(broken, 92, UINT32_MAX, 4);
  reject(Pfs(Pfsc(broken)), "reject excessive cluster heap");
  broken = exfat;
  broken[Heap + 2] ^= 1;
  reject(Pfs(Pfsc(broken)), "reject bad directory checksum");
  broken = exfat;
  Put(broken, Unit + 2 * 4, 2, 4);
  reject(Pfs(Pfsc(broken)), "reject FAT cycle");
  broken = exfat;
  Entry(broken, Heap + 96, u"..", 7, Unit, Unit, true);
  reject(Pfs(Pfsc(broken)), "reject unsafe directory name");
  broken = exfat;
  Entry(broken, Heap + 96, u"EBOOT.BIN", 7, Unit, Unit, false);
  reject(Pfs(Pfsc(broken)), "reject ambiguous member names");
  std::puts("ImageArchiveTests: synthetic tests passed");
}
void Compare(const std::filesystem::path &image,
             const std::filesystem::path &folder) {
  const auto openStart = std::chrono::steady_clock::now();
  auto reader = Common::OpenArchive(image);
  const auto openMs = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - openStart)
                          .count();
  Check(bool(reader), "open real image");
  std::printf("ImageArchiveTests: image opened in %.3f ms\n", openMs);
  const auto root = Common::MakeArchivePath(image);
  uint64_t total = 0;
  size_t files = 0, randomReads = 0;
  std::mt19937_64 random(47);
  std::array<uint8_t, 1 << 20> original{}, compressed{};
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(folder)) {
    if (!entry.is_regular_file())
      continue;
    const auto member = entry.path().lexically_relative(folder);
    Common::File a(entry.path(), Common::File::Mode::Read),
        b(root / member, Common::File::Mode::Read);
    Check(!a.IsInvalid() && !b.IsInvalid() && a.Size() == b.Size(),
          "real image member and size");
    for (uint64_t offset = 0; offset < a.Size();) {
      const auto take = static_cast<uint32_t>(
          std::min<uint64_t>(original.size(), a.Size() - offset));
      uint32_t gotA = 0, gotB = 0;
      a.Read(original.data(), take, &gotA);
      b.Read(compressed.data(), take, &gotB);
      if (gotA != take || gotB != take ||
          std::memcmp(original.data(), compressed.data(), take) != 0) {
        std::fprintf(stderr, "Mismatch: %s at %llu (original %u, image %u)\n",
                     member.string().c_str(),
                     static_cast<unsigned long long>(offset), gotA, gotB);
        std::abort();
      }
      offset += take;
      total += take;
    }
    for (unsigned i = 0; i < 8 && a.Size() != 0; ++i) {
      const auto offset = random() % a.Size();
      const auto length = std::min<uint64_t>(65539, a.Size() - offset);
      Check(Read(a, offset, length) == Read(b, offset, length),
            "random read matches original");
      ++randomReads;
    }
    ++files;
  }
  std::printf("ImageArchiveTests: compared %zu files, %llu bytes, %zu random "
              "reads; all match\n",
              files, static_cast<unsigned long long>(total), randomReads);
}
} // namespace
int main(int argc, char **argv) {
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("kyty_image_test_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Check(std::filesystem::create_directory(dir), "create test directory");
  Synthetic(dir);
  if (argc == 3 && std::string(argv[1]) == "--write-corrupt") {
    Check(std::filesystem::copy_file(dir / "damaged-asset.ffpfsc", argv[2]),
          "export corrupt fixture without overwriting files");
    std::filesystem::remove_all(dir);
    return 0;
  }
  std::filesystem::remove_all(dir);
  if (argc == 3)
    Compare(argv[1], argv[2]);
  return 0;
}
