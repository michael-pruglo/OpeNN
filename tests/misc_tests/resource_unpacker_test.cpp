#include <tests/common/helpers.hpp>
#include <fstream>
#include <misc/resource_unpacker.hpp>

namespace misc
{
    bool operator==(const IdxFileFormat& a, const IdxFileFormat& b)
    {
        return a.magic_number == b.magic_number && a.dimensions == b.dimensions && a.data == b.data;
    }
}

namespace misc::resource_unpacker
{
    using misc::IdxFileFormat;

    const std::string PATH_PREF = "../../tests/misc_tests/";

    template<typename T>
    void write_bytes(std::ofstream& os, T data)
    {
        os.write((char*)&data, sizeof(data));
    }

    void create_idx_file(const std::string& filename, const IdxFileFormat& idx_file)
    {
        std::ofstream outfile(filename, std::ios::out | std::ios::trunc | std::ios::binary);
        assert(outfile);

        write_bytes(outfile, idx_file.magic_number);
        for (const auto& dim: idx_file.dimensions)
            write_bytes(outfile, dim);
        for (const auto& item: idx_file.data)
            write_bytes(outfile, item);

        outfile.close();
    }


    TEST(MiscResourceUnpackerTest, OneDimensional)
    {
        const std::string filename = PATH_PREF + "ru_test_1.idx1-ubyte";
        const std::vector<uint8_t> data{ 67U,65U,78U,65U,82U,89U };
        const IdxFileFormat idx_file{
            .magic_number = 0x00000801U,
            .dimensions = { static_cast<uint32_t>(data.size()) },
            .data = data
        };

        //create_idx_file(filename, idx_file);

        EXPECT_EQ(IdxFileUnpacker::unpack(filename), idx_file);
    }
/*
    TEST(MiscResourceUnpackerTest, TwoDimensional)
    {
        const auto& res = IdxFileUnpacker::unpack("ru_test_2.idx2-ubyte");
        EXPECT_EQ(res, (std::vector<std::vector<uint8_t>>{}));
    }

    TEST(MiscResourceUnpackerTest, ThreeDimensional)
    {
        const auto& res = IdxFileUnpacker::unpack("ru_test_3.idx3-ubyte");
        EXPECT_EQ(res, (std::vector<std::vector<std::vector<uint8_t>>>{}));
    }
    */
}