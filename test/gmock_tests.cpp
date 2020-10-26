#include <gtest/gtest.h>
#include <gmock/gmock.h>   

TEST(GMOCK_tests, int1DArray)
{  
    const int expected_array[2]= {1,2};
    int actual_array[2]= {1,2};
    EXPECT_THAT(expected_array, ::testing::ElementsAreArray(actual_array, 2));
}

TEST(GMOCK_tests, int2DArray)
{  
    const int expected_array[3][4] = {  
    {0, 1, 2, 3} ,   /*  initializers for row indexed by 0 */
    {4, 5, 6, 7} ,   /*  initializers for row indexed by 1 */
    {8, 9, 10, 11}   /*  initializers for row indexed by 2 */
    };

    int actual_array[3][4] = {  
    {0, 1, 2, 3} ,   /*  initializers for row indexed by 0 */
    {4, 5, 6, 7} ,   /*  initializers for row indexed by 1 */
    {8, 9, 10, 11}   /*  initializers for row indexed by 2 */
    };

    //EXPECT_THAT(expected_array, ::testing::ElementsAreArray(actual_array ,3));
}
