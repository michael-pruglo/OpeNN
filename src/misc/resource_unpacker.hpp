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
        static IdxFileFormat read_from_file   (std::ifstream& infile);
        static void          read_magic_number(std::ifstream& infile, IdxFileFormat& ret);
        static void          read_dimensions  (std::ifstream& infile, IdxFileFormat& ret);
        static void          read_data        (std::ifstream& infile, IdxFileFormat& ret);
    };
}