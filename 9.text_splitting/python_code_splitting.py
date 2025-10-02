from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


## eg. Python class for binary search
text = """import math

class BinarySearch:

    def search(self, sorted_list, target):
        # Ensure the list is not empty
        if not sorted_list:
            return -1

        low = 0
        high = len(sorted_list) - 1

        # Loop until the low pointer crosses the high pointer
        while low <= high:
            # Calculate the middle index to avoid potential overflow
            mid = low + (high - low) // 2
            
            # Get the value at the middle index
            mid_val = sorted_list[mid]

            if mid_val == target:
                # Target found, return its index
                return mid
            elif mid_val < target:
                # Middle value is less than target, so target must be in the right half
                low = mid + 1
            else:
                # Middle value is greater than target, so target must be in the left half
                high = mid - 1

        # Target was not found in the list
        return -1

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Instantiate the BinarySearch class
    bs = BinarySearch()

    # 2. Define a sorted list to search through
    my_sorted_list = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    
    # --- Case 1: Target is in the list ---
    target_to_find = 23
    print(f"Searching for: {target_to_find}")
    index = bs.search(my_sorted_list, target_to_find)

    if index != -1:
        print(f"Target {target_to_find} found at index: {index}\n")
    else:
        print(f"Target {target_to_find} not found in the list.\n")
    
    # --- Case 2: Target is NOT in the list ---
    target_to_find_2 = 40
    print(f"Searching for: {target_to_find_2}")
    index_2 = bs.search(my_sorted_list, target_to_find_2)

    if index_2 != -1:
        print(f"Target {target_to_find_2} found at index: {index_2}")
    else:
        print(f"Target {target_to_find_2} not found in the list.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunk = splitter.split_text(text)

print(len(chunk))

print(chunk[1])