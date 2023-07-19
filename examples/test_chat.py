
import os
from pathlib import Path
from typing import Iterator, List

import chatglm_cpp._C as _C

__version__ = "0.2.1"
import time

print("--------")
class Pipeline(_C.Pipeline):
    def __init__(self, model_path: os.PathLike) -> None:
        model_path = Path(model_path)
        super().__init__(str(model_path))

    def stream_chat(
        self,
        history: List[str],
        *,
        max_length: int = 2048,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        num_threads: int = 0,
    ) -> Iterator[str]:
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
        )

        input_ids = self.tokenizer.encode_history(history, max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)

        token_cache = []
        print_len = 0
        while len(output_ids) < max_length:
            next_token_id = self.model.generate_next_token(input_ids, gen_config, n_past, n_ctx)
            n_past += len(input_ids)
            input_ids = [next_token_id]
            output_ids.append(next_token_id)

            token_cache.append(next_token_id)
            output = self.tokenizer.decode(token_cache)

            if output.endswith("\n"):
                yield output[print_len:]
                token_cache = []
                print_len = 0
            elif output.endswith((",", "!", ":", ";", "?", "�")):
                pass
            else:
                yield output[print_len:]
                print_len = len(output)

            if next_token_id == self.model.config.eos_token_id:
                break

        output = self.tokenizer.decode(token_cache)
        yield output[print_len:]

    def chat(
        self,
        history: List[str],
        *,
        max_length: int = 2048,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        num_threads: int = 0,
    ) -> str:
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
        )

        input_ids = self.tokenizer.encode_history(history, max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)
        start = time.time()
        while len(output_ids) < max_length:
            next_token_id = self.model.generate_next_token(input_ids, gen_config, n_past, n_ctx)
            n_past += len(input_ids)
            input_ids = [next_token_id]
            output_ids.append(next_token_id)
            if next_token_id == self.model.config.eos_token_id:
                break
        end = time.time()
        token_count = len(output_ids[n_ctx:])
        print("throughput: {} {}".format((token_count / (end - start)), "tokens/s"))
        output = self.tokenizer.decode(output_ids[n_ctx:])
        return output
    
    
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm2-ggml.bin"
pipeline = Pipeline(DEFAULT_MODEL_PATH)

prompt = "已知信息：\n    以下内容都是提问的设备3045的相关信息:15.照明装置设备整体结构大概可分为 部分组成整机重量 吨为方便设备运输或移动给料斗可升降尾料输送机可折叠侧面楼梯可折叠本章只记录主要零部件如想查阅更详细的零部件信息请参考KJ-3045B 机架图片的链接是: http://img.com/cf9e206b436a624a0f5f89e6f24eab16.png-3序号 10按键可以控制柴油机的速度4.03 设备形态转换当设备移动到指定地点后便可从运输状态图 4.03-1转换为工作状态图 57 -图片的链接是: http://img.com/273bdbe9b0ad0dd0fe687a4c1d0b3261.png 图 4.03-1 设备运输状态图片的链接是: http://img.com/2bf1e565bf2f16f849f65458600ee3fa.png 图 4.03-2 设备工作状态 58 - 尾料输送机工作状态/运输状态装换一拆除运输固定装置拆除尾料输送机运输固定螺栓图 -1序号 1图片的链接是: http://img.com/baab6a53793c8430be0b03272ae3816b.png 1.尾料输送机运输固定螺栓图 -1 尾料输送机运输状态二展开尾料输送机1.展开尾料输送机一段操作遥控器将尾料输送机一段摇杆图 -3缓缓下拨此时尾料输送机一段油7才能使柴油机停止工作手动模式下遥控器无法控制柴油机停止自动模式下只需要按一下遥控器上的柴油机停止按钮-3序号 8柴油机就会减速直至停止如需要更详细的介绍请参照柴油机控制柜使用说明书4.04 设备安装设备安装只能由我公司售后服务工程师或受过专业操作技能培训的人员获得用户授权后方可进行安装作业 \n\n    根据上述已知信息，简洁和专业回答用户的问题，如果问题里询问图片，请返回相关的图片具体链接。如果已知信息里有多个图片，请返回最匹配的图片链接，并用[]包含链接内容而且不要有其他文字描述。\n    如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：3045的运输状态图片\n\n答："
for i in range(8):
    resp = pipeline.chat(
                [prompt],
                max_length=2048,
                max_context_length=512,
                do_sample=False,
                top_k=1,
                top_p=0.9,
                temperature=1.0,
    )

print(resp)