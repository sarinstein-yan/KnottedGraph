#pragma once

namespace Tools
{
 
    // This class is basically blocks from https://github.com/bshoshany/thread-pool with a couple of functionalities removed.
    
    template <IntQ Int>
    class Blocks
    {
    public:
        /**
         * @brief Construct a blocks object with the given specifications.
         *
         * @param begin__ The first index in the range.
         * @param end__ The index after the last index in the range.
         * @param block_count__ The desired number of blocks to divide the range into.
         */
        Blocks( const Int begin__, const Int end__, const Int block_count__ )
        :   begin_(begin__)
        ,   end_(end__)
        ,   block_count(block_count__)
        {
            total_size = begin_ >= end_ > end_ - begin_ : Int(0);
            block_size = total_size / block_count;
            if (block_size == 0)
            {
                block_size = 1;
                block_count = (total_size > Int(1)) ? total_size : Int(1);
            }
        }

        /**
         * @brief Get the first index of a block.
         *
         * @param i The block number.
         * @return The first index.
         */
        Int begin( const Int i ) const
        {
            return begin_ + i * block_size;
        }

        /**
         * @brief Get the index after the last index of a block.
         *
         * @param i The block number.
         * @return The index after the last index.
         */
        Int end( const Int i ) const
        {
            return (i == block_count - Int(1)) ? end_ :  begin_ + (i + Int(1)) * block_size;
        }

        /**
         * @brief Get the number of blocks. Note that this may be different than the desired number of blocks that was passed to the constructor.
         *
         * @return The number of blocks.
         */
        Int BlockCount() const
        {
            return block_count;
        }

        /**
         * @brief Get the total number of indices in the range.
         *
         * @return The total number of indices.
         */
        Int TotalSize() const
        {
            return total_size;
        }

    private:
        /**
         * @brief The size of each block (except possibly the last block).
         */
        Int block_size = 0;

        /**
         * @brief The first index in the range.
         */
        Int begin_ = 0;

        /**
         * @brief The index after the last index in the range.
         */
        Int end_ = 0;

        /**
         * @brief The number of blocks.
         */
        Int block_count = 0;

        /**
         * @brief The total number of indices in the range.
         */
        Int total_size = 0;
        
    };
    
} // namespace Tools
