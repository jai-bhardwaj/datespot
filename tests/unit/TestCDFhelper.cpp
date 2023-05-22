#include <map>
#include <string>
#include <sstream>
#include <unordered_map>

#include <gtest/gtest.h>

#include "CDFhelper.h"

/**
 * @brief Test fixture for the NetCDFhelper class.
 */
class TestNetCDFhelper : public ::testing::Test {
    const static std::map<std::string, unsigned int> validFeatureIndex;

public:
    /**
     * @brief Test case for loading feature index with valid input.
     */
    void TestLoadIndexWithValidInput() {
        std::stringstream inputStream;
        for (const auto& entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_TRUE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_EQ(outputStream.str().find("Error"), std::string::npos);
        ASSERT_EQ(validFeatureIndex.size(), labelsToIndices.size());

        for (const auto& entry : validFeatureIndex) {
            const auto itr = labelsToIndices.find(entry.first);
            ASSERT_NE(itr, labelsToIndices.end());
            ASSERT_EQ(entry.second, itr->second);
        }
    }

    /**
     * @brief Test case for loading feature index with duplicate entry.
     */
    void TestLoadIndexWithDuplicateEntry() {
        std::stringstream inputStream;
        for (const auto& entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        const auto itr = validFeatureIndex.begin();
        inputStream << validFeatureIndex.begin()->first << "\t" << itr->second << "\n";

        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_NE(outputStream.str().find("Error"), std::string::npos);
    }

    /**
     * @brief Test case for loading feature index with duplicate label only.
     */
    void TestLoadIndexWithDuplicateLabelOnly() {
        std::stringstream inputStream;
        for (const auto& entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        inputStream << validFeatureIndex.begin()->first << "\t123\n";

        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_NE(outputStream.str().find("Error"), std::string::npos);
    }

    /**
     * @brief Test case for loading feature index with missing label.
     */
    void TestLoadIndexWithMissingLabel() {
        std::stringstream inputStream;
        inputStream << "\t123\n";
        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_NE(outputStream.str().find("Error"), std::string::npos);
    }

    /**
     * @brief Test case for loading feature index with missing label and tab.
     */
    void TestLoadIndexWithMissingLabelAndTab() {
        std::stringstream inputStream;
        inputStream << "123\n";
        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_NE(outputStream.str().find("Error"), std::string::npos);
    }

    /**
     * @brief Test case for loading feature index with extra tab.
     */
    void TestLoadIndexWithExtraTab() {
        std::stringstream inputStream;
        inputStream << "110510\t123\t121017\n";
        std::unordered_map<std::string, unsigned int> labelsToIndices;
        std::stringstream outputStream;
        ASSERT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
        ASSERT_NE(outputStream.str().find("Error"), std::string::npos);
    }
};

const std::map<std::string, unsigned int> TestNetCDFhelper::validFeatureIndex = {
    { "110510", 26743 },
    { "121019", 26740 },
    { "121017", 26739 },
    { "106401", 26736 },
    { "104307", 26734 }
};

/**
 * @brief Test suite for the TestNetCDFhelper class.
 */
TEST_F(TestNetCDFhelper, TestLoadIndexWithValidInput) {
    TestLoadIndexWithValidInput();
}

TEST_F(TestNetCDFhelper, TestLoadIndexWithDuplicateEntry) {
    TestLoadIndexWithDuplicateEntry();
}

TEST_F(TestNetCDFhelper, TestLoadIndexWithDuplicateLabelOnly) {
    TestLoadIndexWithDuplicateLabelOnly();
}

TEST_F(TestNetCDFhelper, TestLoadIndexWithMissingLabel) {
    TestLoadIndexWithMissingLabel();
}

TEST_F(TestNetCDFhelper, TestLoadIndexWithMissingLabelAndTab) {
    TestLoadIndexWithMissingLabelAndTab();
}

TEST_F(TestNetCDFhelper, TestLoadIndexWithExtraTab) {
    TestLoadIndexWithExtraTab();
}
