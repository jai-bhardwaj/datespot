#ifndef __GPUSORT_H__
#define __GPUSORT_H__

#include <memory>
#include <span>

/**
 * @brief Templated class for GPU sorting of key-value pairs.
 * 
 * @tparam KeyType The type of the keys.
 * @tparam ValueType The type of the values.
 */
template<typename KeyType, typename ValueType>
class GpuSort {
private:
    const unsigned int _items;
    const unsigned int _itemStride;
    std::unique_ptr<GpuBuffer<KeyType>> _pbKey;
    std::unique_ptr<GpuBuffer<ValueType>> _pbValue;
    size_t _tempBytes;
    std::unique_ptr<GpuBuffer<char>> _pbTemp;
    std::span<KeyType> _keySpan;
    std::span<ValueType> _valueSpan;

public:
    /**
     * @brief Constructs a GpuSort object.
     * 
     * @param items The total number of items to be sorted.
     */
    GpuSort(unsigned int items)
        : _items(items),
          _itemStride(((items + 511) >> 9) << 9),
          _pbKey(std::make_unique<GpuBuffer<KeyType>>(_itemStride * 2)),
          _pbValue(std::make_unique<GpuBuffer<ValueType>>(_itemStride * 2)),
          _tempBytes(kInitSort(_items, _pbValue.get(), _pbKey.get())),
          _pbTemp(std::make_unique<GpuBuffer<char>>(_tempBytes)),
          _keySpan(_pbKey->get(), _itemStride * 2),
          _valueSpan(_pbValue->get(), _itemStride * 2) {}

    /**
     * @brief Sorts the key-value pairs using GPU.
     * 
     * @return true if the sorting is successful, false otherwise.
     */
    [[nodiscard]] bool Sort() {
        return kSort(
            _items, _keySpan.subspan(0, _items).data(), _keySpan.subspan(_itemStride, _items).data(),
            _valueSpan.subspan(0, _items).data(), _valueSpan.subspan(_itemStride, _items).data(),
            _pbTemp->get(), _tempBytes);
    }

    /**
     * @brief Returns a span representing the key array.
     * 
     * @return std::span<KeyType> The span representing the key array.
     */
    std::span<KeyType> GetKeySpan() {
        return _keySpan.subspan(0, _items);
    }

    /**
     * @brief Returns a span representing the value array.
     * 
     * @return std::span<ValueType> The span representing the value array.
     */
    std::span<ValueType> GetValueSpan() {
        return _valueSpan.subspan(0, _items);
    }
};
#endif
