// Image layout and validation adapted from KytyPlus GameImage.cs (MIT).
// Copyright and license: LICENSES/KytyPlus-GameImage-MIT.txt.
#include "common/archiveReader.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <zlib.h>

namespace Common {
namespace {

using Bytes                     = std::vector<uint8_t>;
constexpr uint64_t MaxMetadata  = 128u << 20u;
constexpr uint64_t MaxDirectory = 16u << 20u;
constexpr uint64_t MaxEntries   = 1000000;

bool Range(uint64_t total, uint64_t offset, uint64_t length) {
	return offset <= total && length <= total - offset;
}
uint64_t Blocks(uint64_t size, uint64_t unit) {
	return size == 0 ? 0 : (size - 1) / unit + 1;
}
uint64_t Little(std::span<const uint8_t> bytes, size_t offset, size_t size) {
	uint64_t value = 0;
	for (size_t i = 0; i < size; ++i)
		value |= uint64_t(bytes[offset + i]) << (i * 8);
	return value;
}
uint16_t U16(std::span<const uint8_t> b, size_t p) {
	return Little(b, p, 2);
}
uint32_t U32(std::span<const uint8_t> b, size_t p) {
	return Little(b, p, 4);
}
uint64_t U64(std::span<const uint8_t> b, size_t p) {
	return Little(b, p, 8);
}
bool PowerOfTwo(uint64_t value) {
	return value != 0 && (value & (value - 1)) == 0;
}

class Image {
public:
	virtual ~Image()                                              = default;
	uint64_t     size                                             = 0;
	virtual bool Read(uint64_t offset, std::span<uint8_t> output) = 0;
	Bytes        Get(uint64_t offset, uint64_t length) {
		if (length > MaxMetadata || !Range(size, offset, length)) return {};
		Bytes bytes(static_cast<size_t>(length));
		return Read(offset, bytes) ? std::move(bytes) : Bytes {};
	}
};
using ImagePtr = std::shared_ptr<Image>;

class HostImage final: public Image {
public:
	explicit HostImage(const std::filesystem::path& path): file(path, File::Mode::Read) {
		if (!file.IsInvalid()) size = file.Size();
	}
	bool Read(uint64_t offset, std::span<uint8_t> output) override {
		if (!Range(size, offset, output.size()) || output.size() > UINT32_MAX || file.IsInvalid() ||
		    !file.Seek(offset))
			return false;
		uint32_t read = 0;
		file.Read(output.data(), static_cast<uint32_t>(output.size()), &read);
		return read == output.size();
	}

private:
	File file;
};

class SliceImage final: public Image {
public:
	SliceImage(ImagePtr source, uint64_t start, uint64_t length)
	    : source(std::move(source)), start(start) {
		size = length;
	}
	bool Read(uint64_t offset, std::span<uint8_t> output) override {
		return Range(size, offset, output.size()) && source->Read(start + offset, output);
	}

private:
	ImagePtr source;
	uint64_t start;
};
ImagePtr Slice(const ImagePtr& source, uint64_t start, uint64_t length) {
	return Range(source->size, start, length) ? std::make_shared<SliceImage>(source, start, length)
	                                          : nullptr;
}

class ChainImage final: public Image {
public:
	ChainImage(ImagePtr source, uint64_t unit, std::vector<uint64_t> offsets, uint64_t length)
	    : source(std::move(source)), unit(unit), offsets(std::move(offsets)) {
		size = length;
	}
	bool Read(uint64_t offset, std::span<uint8_t> output) override {
		if (!Range(size, offset, output.size())) return false;
		while (!output.empty()) {
			const auto index = offset / unit, inside = offset % unit;
			const auto take = std::min<uint64_t>(unit - inside, output.size());
			if (index >= offsets.size() ||
			    !source->Read(offsets[index] + inside, output.first(take)))
				return false;
			offset += take;
			output = output.subspan(take);
		}
		return true;
	}

private:
	ImagePtr              source;
	uint64_t              unit;
	std::vector<uint64_t> offsets;
};

class ZeroTailImage final: public Image {
public:
	ZeroTailImage(ImagePtr source, uint64_t valid): source(std::move(source)), valid(valid) {
		size = this->source->size;
	}
	bool Read(uint64_t offset, std::span<uint8_t> output) override {
		if (!Range(size, offset, output.size())) return false;
		const auto take = offset < valid ? std::min<uint64_t>(output.size(), valid - offset) : 0;
		if (take != 0 && !source->Read(offset, output.first(take))) return false;
		std::fill(output.begin() + take, output.end(), 0);
		return true;
	}

private:
	ImagePtr source;
	uint64_t valid;
};

class PfscImage final: public Image {
public:
	static ImagePtr Open(ImagePtr source) {
		auto result    = std::make_shared<PfscImage>();
		result->source = std::move(source);
		const auto h   = result->source->Get(0, 0x30);
		if (h.empty() || std::memcmp(h.data(), "PFSC", 4) != 0) return {};
		result->unit = U32(h, 12);
		result->size = U64(h, 40);
		if (!PowerOfTwo(result->unit) || result->unit < 4096 || result->unit > (1u << 20u) ||
		    U64(h, 16) != result->unit || result->size == 0)
			return {};
		const auto count = Blocks(result->size, result->unit);
		if (count > 16000000) return {};
		const auto tableStart = U64(h, 24), dataStart = U64(h, 32);
		const auto tableSize = (count + 1) * 8;
		if (tableStart < h.size() || !Range(dataStart, tableStart, tableSize)) return {};
		const auto table = result->source->Get(tableStart, tableSize);
		if (table.empty()) return {};
		result->offsets.reserve(count + 1);
		for (uint64_t i = 0; i <= count; ++i) {
			const auto offset = U64(table, i * 8);
			if (offset > result->source->size || (i == 0 && offset != dataStart) ||
			    (i != 0 && (offset <= result->offsets.back() ||
			                offset - result->offsets.back() > result->unit)))
				return {};
			result->offsets.push_back(offset);
		}
		return result;
	}
	bool Read(uint64_t offset, std::span<uint8_t> output) override {
		if (!Range(size, offset, output.size())) return false;
		while (!output.empty()) {
			const auto index = offset / unit, inside = offset % unit;
			auto       entry = cache.find(index);
			if (entry == cache.end()) {
				auto stored = source->Get(offsets[index], offsets[index + 1] - offsets[index]);
				if (stored.empty()) return false;
				Bytes decoded(unit);
				if (stored.size() == unit)
					decoded = std::move(stored);
				else {
					z_stream stream {};
					stream.next_in   = stored.data();
					stream.avail_in  = stored.size();
					stream.next_out  = decoded.data();
					stream.avail_out = decoded.size();
					if (inflateInit(&stream) != Z_OK) return false;
					const auto status = inflate(&stream, Z_FINISH);
					const bool valid  = status == Z_STREAM_END && stream.total_out == unit &&
					                    stream.total_in == stored.size();
					inflateEnd(&stream);
					if (!valid) return false;
				}
				if (cache.size() == 16) {
					cache.erase(order.front());
					order.pop_front();
				}
				entry = cache.emplace(index, std::move(decoded)).first;
				order.push_back(index);
			}
			const auto take = std::min<uint64_t>(unit - inside, output.size());
			std::memcpy(output.data(), entry->second.data() + inside, take);
			offset += take;
			output = output.subspan(take);
		}
		return true;
	}

private:
	ImagePtr                            source;
	uint64_t                            unit = 0;
	std::vector<uint64_t>               offsets;
	std::unordered_map<uint64_t, Bytes> cache;
	std::deque<uint64_t>                order;
};

bool SafeName(std::string_view name) {
	if (name.empty() || name == "." || name == "..") return false;
	return std::none_of(name.begin(), name.end(), [](unsigned char c) {
		return c < 32 || c == '/' || c == '\\' || c == ':';
	});
}
std::string Key(std::string_view name) {
	std::string result(name);
	for (auto& c: result)
		if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
	return result;
}
struct Member {
	std::string path;
	ImagePtr    data;
};
using Members = std::vector<Member>;

class PfsTree {
public:
	explicit PfsTree(ImagePtr image): image(std::move(image)) {}
	bool Parse(Members& files) {
		const auto h = image->Get(0, 0x380);
		if (h.empty() || (U64(h, 0) != 1 && U64(h, 0) != 2) || U64(h, 8) != 20130315) return false;
		const auto mode = U16(h, 0x1c);
		if ((mode & 4) != 0) return false; // Encrypted standalone images are not supported.
		signedNodes = (mode & 1) != 0;
		wide        = (mode & 2) != 0;
		if (wide && !signedNodes) return false;
		const uint64_t stride = signedNodes ? (wide ? 0x310 : 0x2c8) : 0xa8;
		unit                  = U32(h, 0x20);
		if (!PowerOfTwo(unit) || unit < 4096 || unit > (1u << 20u) || unit < stride) return false;
		const auto count = U64(h, 0x30), perBlock = unit / stride;
		if (count == 0 || count > MaxEntries) return false;
		const auto needed = Blocks(count, perBlock);
		if (U64(h, 0x40) < needed || needed > MaxMetadata / unit) return false;
		const auto table = image->Get(unit, needed * unit);
		if (table.empty()) return false;
		for (uint64_t i = 0; i < count; ++i) {
			const auto begin = table.begin() + (i / perBlock * unit + i % perBlock * stride);
			Bytes      record(begin, begin + stride);
			nodes.push_back({U16(record, 0), U32(record, 4), U64(record, 8), U64(record, 16),
			                 wide ? U64(record, 0x60) : U32(record, 0x60), std::move(record)});
		}
		std::vector<DirEntry> root;
		if (!Directory(0, root)) return false;
		uint32_t first = 0;
		for (const auto& entry: root)
			if (entry.name == "uroot" && entry.type == 3) first = entry.inode;
		return Walk(first, "", 0, files);
	}

private:
	struct Node {
		uint16_t mode;
		uint32_t flags;
		uint64_t size, logical, blocks;
		Bytes    record;
	};
	struct DirEntry {
		uint32_t    inode, type;
		std::string name;
	};
	uint64_t Pointer(std::span<const uint8_t> record, size_t offset) const {
		return wide ? U64(record, offset) : U32(record, offset);
	}
	bool Indirect(uint64_t block, unsigned depth, uint64_t needed, std::vector<uint64_t>& chain,
	              std::unordered_set<uint64_t>& seen) {
		if (depth == 0 || depth > 5 || block == 0 || block >= image->size / unit ||
		    !seen.insert(block).second)
			return false;
		const auto bytes = image->Get(block * unit, unit);
		if (bytes.empty()) return false;
		const size_t width = wide ? 8 : 4, step = width + (signedNodes ? 32 : 0);
		for (size_t p = 0; p + step <= bytes.size() && chain.size() < needed; p += step) {
			const auto next = Pointer(bytes, p + (signedNodes ? 32 : 0));
			if (depth == 1) {
				if (next == 0 || next >= image->size / unit) return false;
				chain.push_back(next * unit);
			} else if (!Indirect(next, depth - 1, needed, chain, seen))
				return false;
		}
		return true;
	}
	ImagePtr Data(uint32_t inode) {
		if (inode >= nodes.size()) return {};
		const auto& node = nodes[inode];
		if (node.size == 0) return Slice(image, 0, 0);
		if (node.blocks == 0 || node.blocks > image->size / unit || node.blocks > 16000000 ||
		    Blocks(node.size, unit) > node.blocks)
			return {};
		const size_t width = wide ? 8 : 4, step = width + (signedNodes ? 32 : 0);
		const size_t ptr      = (wide ? 0x68 : 0x64) + (signedNodes ? 32 : 0);
		const auto   first    = Pointer(node.record, ptr);
		const auto   sentinel = wide ? UINT64_MAX : UINT32_MAX;
		ImagePtr     stored;
		if (node.blocks <= 1 || Pointer(node.record, ptr + step) == sentinel) {
			if (first == 0 || first > image->size / unit ||
			    node.blocks > image->size / unit - first)
				return {};
			stored = Slice(image, first * unit, node.size);
		} else {
			std::vector<uint64_t> chain;
			chain.reserve(node.blocks);
			for (size_t i = 0; i < 12 && chain.size() < node.blocks; ++i) {
				const auto block = Pointer(node.record, ptr + i * step);
				if (block == 0 || block >= image->size / unit) return {};
				chain.push_back(block * unit);
			}
			std::unordered_set<uint64_t> seen;
			for (unsigned i = 0; i < 5 && chain.size() < node.blocks; ++i)
				if (!Indirect(Pointer(node.record, ptr + (12 + i) * step), i + 1, node.blocks,
				              chain, seen))
					return {};
			if (chain.size() != node.blocks) return {};
			stored = std::make_shared<ChainImage>(image, unit, std::move(chain), node.size);
		}
		if (!stored || (node.flags & 1) == 0) return stored;
		auto decoded = PfscImage::Open(stored);
		return decoded && node.logical != 0 ? Slice(decoded, 0, node.logical) : nullptr;
	}
	bool Directory(uint32_t inode, std::vector<DirEntry>& entries) {
		if (inode >= nodes.size() || (nodes[inode].mode & 0xf000) != 0x4000 ||
		    nodes[inode].size > MaxDirectory)
			return false;
		auto source = Data(inode);
		if (!source) return false;
		const auto bytes = source->Get(0, source->size);
		if (bytes.size() != source->size) return false;
		for (size_t p = 0; p + 16 <= bytes.size();) {
			const auto child = U32(bytes, p), type = U32(bytes, p + 4);
			const auto names = U32(bytes, p + 8), length = U32(bytes, p + 12);
			if (length == 0) break;
			if (length < 16 || length > bytes.size() - p || names == 0 || names > length - 16 ||
			    child >= nodes.size())
				return false;
			if (type == 2 || type == 3) {
				std::string name(reinterpret_cast<const char*>(bytes.data() + p + 16), names);
				while (!name.empty() && name.back() == '\0')
					name.pop_back();
				if (!SafeName(name)) return false;
				entries.push_back({child, type, std::move(name)});
			} else if (type != 4 && type != 5)
				return false;
			p += length;
		}
		return true;
	}
	bool Walk(uint32_t inode, const std::string& prefix, unsigned depth, Members& files) {
		if (depth > 64 || !visited.insert(inode).second) return false;
		std::vector<DirEntry> entries;
		if (!Directory(inode, entries)) return false;
		for (const auto& entry: entries) {
			const auto name = prefix + entry.name;
			if (entry.type == 3) {
				if (files.size() >= MaxEntries) return false;
				files.push_back({name + "/", {}});
				if (!Walk(entry.inode, name + "/", depth + 1, files)) return false;
			} else {
				if ((nodes[entry.inode].mode & 0xf000) != 0x8000 || files.size() >= MaxEntries)
					return false;
				auto data = Data(entry.inode);
				if (!data) return false;
				files.push_back({name, std::move(data)});
			}
		}
		return true;
	}
	ImagePtr                     image;
	uint64_t                     unit        = 0;
	bool                         signedNodes = false, wide = false;
	std::vector<Node>            nodes;
	std::unordered_set<uint32_t> visited;
};

bool Utf16Name(std::span<const uint8_t> bytes, size_t length, std::string& name) {
	if (length == 0 || length > bytes.size() / 2) return false;
	for (size_t i = 0; i < length; ++i) {
		uint32_t c = U16(bytes, i * 2);
		if (c >= 0xd800 && c <= 0xdbff) {
			if (++i >= length) return false;
			const auto low = U16(bytes, i * 2);
			if (low < 0xdc00 || low > 0xdfff) return false;
			c = 0x10000 + ((c - 0xd800) << 10) + low - 0xdc00;
		} else if (c >= 0xdc00 && c <= 0xdfff)
			return false;
		if (c < 0x80)
			name += static_cast<char>(c);
		else {
			if (c >= 0x10000) name += static_cast<char>(0xf0 | (c >> 18));
			if (c >= 0x800)
				name += static_cast<char>((c >= 0x10000 ? 0x80 : 0xe0) | ((c >> 12) & 0x3f));
			name += static_cast<char>((c >= 0x800 ? 0x80 : 0xc0) | ((c >> 6) & 0x3f));
			name += static_cast<char>(0x80 | (c & 0x3f));
		}
	}
	return SafeName(name);
}

class ExfatTree {
public:
	explicit ExfatTree(ImagePtr image): image(std::move(image)) {}
	bool Parse(Members& files) {
		const auto h = image->Get(0, 512);
		if (h.empty() || std::memcmp(h.data() + 3, "EXFAT   ", 8) != 0 || h[510] != 0x55 ||
		    h[511] != 0xaa || h[108] < 9 || h[108] > 12 || unsigned(h[108]) + h[109] > 25 ||
		    h[110] != 1)
			return false;
		const uint64_t sector = uint64_t(1) << h[108];
		unit                  = uint64_t(1) << (unsigned(h[108]) + h[109]);
		fat                   = uint64_t(U32(h, 80)) * sector;
		heap                  = uint64_t(U32(h, 88)) * sector;
		clusters              = U32(h, 92);
		if (clusters == 0 || U64(h, 72) > image->size / sector ||
		    !Range(image->size, fat, uint64_t(U32(h, 84)) * sector) ||
		    !Range(image->size, heap, uint64_t(clusters) * unit) ||
		    (uint64_t(clusters) + 2) * 4 > uint64_t(U32(h, 84)) * sector)
			return false;
		return Walk(U32(h, 96), 0, false, "", 0, files);
	}

private:
	bool     ValidCluster(uint32_t c) const { return c >= 2 && uint64_t(c) - 2 < clusters; }
	uint64_t Offset(uint32_t c) const { return heap + (uint64_t(c) - 2) * unit; }
	ImagePtr Data(uint32_t first, uint64_t length, bool contiguous, uint64_t valid,
	              bool root = false) {
		if (valid > length) return {};
		if (length == 0 && !root) return Slice(image, 0, 0);
		if (!ValidCluster(first)) return {};
		const auto needed = Blocks(length, unit);
		ImagePtr   result;
		if (contiguous) {
			if (root || needed > uint64_t(clusters) - (first - 2)) return {};
			result = Slice(image, Offset(first), length);
		} else {
			if (needed > clusters || needed > 16000000) return {};
			std::vector<uint64_t>        chain;
			std::unordered_set<uint32_t> seen;
			uint32_t                     c = first;
			while (root || chain.size() < needed) {
				if (!ValidCluster(c) || !seen.insert(c).second ||
				    (root && chain.size() >= MaxDirectory / unit))
					return {};
				chain.push_back(Offset(c));
				const auto next = image->Get(fat + uint64_t(c) * 4, 4);
				if (next.empty()) return {};
				c = U32(next, 0);
				if (c >= 0xfffffff8) {
					if (!root && chain.size() != needed) return {};
					break;
				}
			}
			if (root) {
				length = chain.size() * unit;
				valid  = length;
			}
			result = std::make_shared<ChainImage>(image, unit, std::move(chain), length);
		}
		return result ? std::make_shared<ZeroTailImage>(std::move(result), valid) : nullptr;
	}
	bool Walk(uint32_t first, uint64_t length, bool contiguous, const std::string& prefix,
	          unsigned depth, Members& files) {
		if (depth > 64 || length > MaxDirectory || !visited.insert(first).second) return false;
		auto source = Data(first, length, contiguous, length, depth == 0);
		if (!source) return false;
		const auto bytes = source->Get(0, source->size);
		if (bytes.size() != source->size) return false;
		for (size_t p = 0; p + 32 <= bytes.size();) {
			const auto type = bytes[p];
			if (type == 0) break;
			if (type != 0x85) {
				if ((type & 0x80) != 0 && type != 0x81 && type != 0x82 && type != 0x83 &&
				    (type & 0x20) == 0)
					return false;
				p += 32;
				continue;
			}
			const auto   secondary = bytes[p + 1];
			const size_t total     = (size_t(secondary) + 1) * 32;
			if (secondary < 2 || total > bytes.size() - p || bytes[p + 32] != 0xc0) return false;
			uint16_t checksum = 0;
			for (size_t i = 0; i < total; ++i)
				if (i != 2 && i != 3)
					checksum = uint16_t(((checksum << 15) | (checksum >> 1)) + bytes[p + i]);
			if (checksum != U16(bytes, p + 2)) return false;
			Bytes nameBytes;
			for (unsigned i = 2; i <= secondary; ++i) {
				const auto q = p + i * 32;
				if (bytes[q] != 0xc1) return false;
				nameBytes.insert(nameBytes.end(), bytes.begin() + q + 2, bytes.begin() + q + 32);
			}
			std::string name;
			if (!Utf16Name(nameBytes, bytes[p + 35], name)) return false;
			const auto size = U64(bytes, p + 56), valid = U64(bytes, p + 40);
			const auto cluster = U32(bytes, p + 52);
			const bool noFat   = (bytes[p + 33] & 2) != 0;
			if ((U16(bytes, p + 4) & 0x10) != 0) {
				if (files.size() >= MaxEntries) return false;
				files.push_back({prefix + name + "/", {}});
				if (valid != size ||
				    !Walk(cluster, size, noFat, prefix + name + "/", depth + 1, files))
					return false;
			} else {
				if (files.size() >= MaxEntries) return false;
				auto data = Data(cluster, size, noFat, valid);
				if (!data) return false;
				files.push_back({prefix + name, std::move(data)});
			}
			p += total;
		}
		return true;
	}
	ImagePtr                     image;
	uint64_t                     fat = 0, heap = 0, unit = 0;
	uint32_t                     clusters = 0;
	std::unordered_set<uint32_t> visited;
};

bool ReadTree(ImagePtr image, Members& files, unsigned depth = 0) {
	if (depth > 4) return false;
	const auto header = image->Get(0, 16);
	if (header.empty()) return false;
	if (std::memcmp(header.data(), "PFSC", 4) == 0) {
		auto decoded = PfscImage::Open(std::move(image));
		return decoded && ReadTree(std::move(decoded), files, depth + 1);
	}
	if (std::memcmp(header.data() + 3, "EXFAT   ", 8) == 0)
		return ExfatTree(std::move(image)).Parse(files);
	Members outer;
	if (!PfsTree(std::move(image)).Parse(outer)) return false;
	ImagePtr inner;
	for (const auto& entry: outer) {
		const auto name = Key(entry.path);
		if (name == "pfs_image.dat" || name.ends_with(".exfat") || name.ends_with(".ffpkg") ||
		    name.ends_with(".ffpfs")) {
			if (inner) return false;
			inner = entry.data;
		}
	}
	if (inner) return ReadTree(std::move(inner), files, depth + 1);
	files = std::move(outer);
	return true;
}

class ImageArchive final: public ArchiveReader {
public:
	bool Open(const std::filesystem::path& path) {
		Members files;
		if (!ReadTree(std::make_shared<HostImage>(path), files)) return false;
		nodes.push_back({{}, {}, {}});
		lookup.emplace("", 0);
		for (auto& file: files) {
			uint64_t parent = 0;
			size_t   begin  = 0;
			while (begin < file.path.size()) {
				const auto slash  = file.path.find('/', begin);
				const bool isFile = slash == std::string::npos;
				const auto end    = isFile ? file.path.size() : slash;
				const auto name   = file.path.substr(begin, end - begin);
				if (!SafeName(name)) return false;
				const auto key   = Key(file.path.substr(0, end));
				auto       found = lookup.find(key);
				uint64_t   id;
				if (found != lookup.end()) {
					id = found->second;
					if (isFile || nodes[id].data) return false;
				} else {
					if (nodes.size() >= MaxEntries * 2) return false;
					id = nodes.size();
					nodes.push_back({name, isFile ? std::move(file.data) : ImagePtr {}, {}});
					lookup.emplace(key, id);
					nodes[parent].children.push_back(id);
				}
				parent = id;
				if (isFile) break;
				begin = slash + 1;
			}
		}
		return lookup.contains("eboot.bin");
	}
	std::optional<Entry> Find(std::string_view member) override {
		const auto found = lookup.find(Key(member));
		if (found == lookup.end()) return {};
		const auto id = found->second;
		return Entry {id, nodes[id].data ? nodes[id].data->size : 0, bool(nodes[id].data)};
	}
	std::vector<File::DirEntry> List(std::string_view member) override {
		const auto found = lookup.find(Key(member));
		if (found == lookup.end() || nodes[found->second].data) return {};
		std::vector<File::DirEntry> result;
		for (const auto id: nodes[found->second].children)
			result.push_back({nodes[id].name, bool(nodes[id].data)});
		return result;
	}
	uint64_t Read(uint64_t id, uint64_t offset, uint32_t length, void* data) override {
		if (id >= nodes.size() || !nodes[id].data || data == nullptr) return 0;
		auto& source = nodes[id].data;
		if (offset >= source->size) return 0;
		const auto take = std::min<uint64_t>(length, source->size - offset);
		// Metadata is immutable. Serialize access to host cursors and the decompression caches.
		std::lock_guard lock(mutex);
		return source->Read(offset, {static_cast<uint8_t*>(data), static_cast<size_t>(take)}) ? take
		                                                                                      : 0;
	}

private:
	struct Node {
		std::string           name;
		ImagePtr              data;
		std::vector<uint64_t> children;
	};
	std::vector<Node>                         nodes;
	std::unordered_map<std::string, uint64_t> lookup;
	std::mutex                                mutex;
};

} // namespace

std::shared_ptr<ArchiveReader> OpenImageArchive(const std::filesystem::path& path) {
	auto reader = std::make_shared<ImageArchive>();
	return reader->Open(path) ? reader : nullptr;
}

} // namespace Common
