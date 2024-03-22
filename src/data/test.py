class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        if n == 0:
            return nums1
        i,ii=0,0
        t_nums = nums1[:]
        while i+ii < len(nums1):
            v1 = t_nums[i]
            if ii < n:
                v2 = nums2[ii]
            else:
                v2 = float('inf')
            if v1 < v2 and i < m:
                nums1[i+ii]=v1
                i+=1
            else:
                nums1[i+ii]=v2
                ii+=1
        return nums1

s = Solution()
print(s.merge([1,2,3,0,0,0],3,[2,5,6],3))
print(s.merge([2,0],2,[1],1))