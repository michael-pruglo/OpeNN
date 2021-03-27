#pragma once

#include <vector>
#include <fstream>

namespace misc
{
    template<typename T>
    struct IdxFileGeneralFormat
    {
        uint32_t magic_number;
        std::vector<uint32_t> dimensions;
        std::vector<T> data;
    };

    using IdxFileFormat = IdxFileGeneralFormat<uint8_t>;

    class IdxFileUnpacker
    {
    public:
        static IdxFileFormat unpack(const std::string& filename);

    private:
        static IdxFileFormat read_from_file(std::ifstream& infile);
    };
}