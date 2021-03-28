#include <misc/resource_unpacker.hpp>
#include <cassert>
#include <numeric>

misc::IdxFileFormat misc::IdxFileUnpacker::unpack(const std::string& filename)
{
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    assert(infile);

    auto ret = read_from_file(infile);

    infile.close();
    return ret;
}

misc::IdxFileFormat misc::IdxFileUnpacker::read_from_file(std::ifstream& infile)
{
    IdxFileFormat ret;

    read_magic_number(infile, ret);
    read_dimensions(infile, ret);
    read_data(infile, ret);

    return ret;
}

void misc::IdxFileUnpacker::read_magic_number(std::ifstream& infile, IdxFileFormat& ret)
{
    read_bytes(infile, ret.magic_number);

    uint8_t zeros = (ret.magic_number & 0xFFFF0000u) >> 16u;
    assert(zeros == 0);

    uint8_t data_type_indicator = (ret.magic_number & 0x0000FF00u) >> 8u;
    assert(data_type_indicator == 0x08 && "only 0x08 is supported atm");
}

void misc::IdxFileUnpacker::read_dimensions(std::ifstream& infile, IdxFileFormat& ret)
{
    uint8_t dimensions_amount = ret.magic_number & 0x000000FFu;
    ret.dimensions.resize(dimensions_amount);
    for (uint8_t i = 0; i < dimensions_amount; ++i)
        read_bytes(infile, ret.dimensions[i]);
}

void misc::IdxFileUnpacker::read_data(std::ifstream& infile, IdxFileFormat& ret)
{
    size_t n = 0;
    const size_t expected_data_items =
        std::accumulate(ret.dimensions.begin(), ret.dimensions.end(), 1, std::multiplies());
    ret.data.resize(expected_data_items);
    while (true)
    {
        read_bytes(infile, ret.data[n]);
        if (infile.eof())
            break;
        else
            ++n;
    }

    assert(expected_data_items == n);
}
