from utils import get_logger

def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def main():

    logger = get_logger("Test")
    
    # 使用 logger 输出 helloworld (保留之前的测试逻辑)
    logger.info("helloworld")

    # 冒泡排序测试
    test_list = [64, 34, 25, 12, 22, 11, 90]
    logger.info(f"原始列表: {test_list}")
    sorted_list = bubble_sort(test_list.copy()) # 复制列表以避免修改原始列表
    logger.info(f"排序后列表: {sorted_list}")

if __name__ == "__main__":
    main()
