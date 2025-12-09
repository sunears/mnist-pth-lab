from utils import get_logger
from torch import nn
from torch import Tensor
def main():
    linear = nn.Linear(3, 5)
    linear.state_dict()   # 只是触发一下参数初始化
    
    logger = get_logger("Test")
    
    # 先打印一个空行，让输出更清爽
    logger.info("")  # 或者 logger.info("\n")
    
    # 带颜色的关键一行（绿色 + 加粗）
    logger.info(
        "\033[1;32mLinear layer state_dict:\033[0m\n%s",
        linear.state_dict()
    )
    
    # input=Tensor([[1.0, 2.0, 3.0]])
    # output=linear(input)
    # logger.info("\033[1;32mLinear layer input:\033[0m")
    # logger.info("input: \n%s", input)
    # logger.info("\033[1;32mLinear layer output:\033[0m")
    # logger.info("output: \n%s", output)
    
    # input=Tensor([
    #     [0.5, 1.5, -2.0],
    #     [3.0, -0.5, 2.0]
    # ])
    # output=linear(input)
    # logger.info("\033[1;32mLinear layer new input:\033[0m")
    # logger.info("input: \n%s", input)
    # logger.info("\033[1;32mLinear layer new output:\033[0m")
    # logger.info("output: \n%s", output)

    input=Tensor([
        [
            [1.0, 2.0, 3.0],
            [4, 5, 6],
        ],  
        [
            [11, 22, 33],
            [44, 55, 66],
        ],
    ])
    output=linear(input)
    logger.info("\033[1;32mLinear layer 3D input shape:\033[0m")
    logger.info(input.shape)
    logger.info("\033[1;32mLinear layer 3D input:\033[0m")
    logger.info("input: \n%s", input)
    logger.info("\033[1;32mLinear layer 3D output:\033[0m")
    logger.info("output: \n%s", output)
    logger.info("\033[1;32mLinear layer 3D output shape:\033[0m")
    logger.info(output.shape)
if __name__ == "__main__":
    main()