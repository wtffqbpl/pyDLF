T& operator[](size_t index) {
    return data_[index];
}

const T& operator[](size_t index) const {
    return data_[index];
}

T* data() { return data_.data(); }

const T* data() const { return data_.data(); } 