#pragma once

#include <vector>
#include <fstream>

namespace misc
{
    template<typename T>
    void read_bytes(std::ifstream& in, T& data) //reverse bytes to flip to big-endian
    {
        auto* const addr = (char*)&data;
        const size_t len = sizeof(data);
        in.read(addr, len);
        std::reverse(addr, addr+len);
    }

    template<typename T>
    void write_bytes(std::ofstream& os, T data) //reverse bytes to flip to big-endian
    {
        auto* const addr = (char*)&data;
        const size_t len = sizeof(data);
        std::reverse(addr, addr+len);
        os.write(addr, len);
    }

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