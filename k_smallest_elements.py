def partition(arr, start, end):
    """
    Partition function for quickSort(). It uses the first element as the pivot.
    """
    j = start
    pivot = start
 
    for i in range(start+1, end):
        if arr[i]<arr[pivot]:
            j = j+1
            # swap elements i and j
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
    if j<end:
        # swap elements j and pivot
        temp = arr[pivot]
        arr[pivot] = arr[j]
        arr[j] = temp
 
    # return index of pivot in the partitioned array 
    return j
 
def quickSort(arr, start, end, k):
    """
    Quick sort function to partition the k smallest elements in arr in place, 
    assuming that elements from 0 to start-1 have been correctly partitioned already.
    """
    if start < end:
        # build partition using the first element as pivot
        p = partition(arr, start, end)

        # if pivot is the kth element, then return
        if p == k-1:
            return
        
        # if pivot is larger than the kth, partition agian from start to p
        if p > k-1:
            quickSort(arr, start,p, k)
        # if pivot is smaller than the kth, partition agian from p to end
        else:
            quickSort(arr, p+1, end, k)


if __name__ == "__main__":
    arr = [0,4,11,15,1,5,7,8,3,10,9,6]
    k = 5
 
    quickSort(arr, 0, len(arr), k);
 
    print(arr)