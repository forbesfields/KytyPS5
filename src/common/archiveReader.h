#ifndef KYTY_COMMON_ARCHIVE_READER_H_
#define KYTY_COMMON_ARCHIVE_READER_H_

#include "common/file.h"

#include <memory>
#include <optional>
#include <string_view>

namespace Common {

// Backends share virtual paths, reader lifetimes and the host-stack IO worker.
class ArchiveReader {
public:
	struct Entry {
		uint64_t id;
		uint64_t size;
		bool     is_file;
	};

	virtual ~ArchiveReader()                                                       = default;
	virtual std::optional<Entry>        Find(std::string_view member)              = 0;
	virtual std::vector<File::DirEntry> List(std::string_view member)              = 0;
	virtual uint64_t Read(uint64_t id, uint64_t offset, uint32_t size, void* data) = 0;
};

[[nodiscard]] std::shared_ptr<ArchiveReader> OpenImageArchive(const std::filesystem::path& path);

} // namespace Common

#endif
