import torch

def add_hook(model):
    total_feat_out = []
    total_feat_in = []

    def hook_fn_forward(module, input, output):
        # print(module)  # 用于区分模块
        # print('input', input[0].shape)  # 首先打印出来
        # print('output', output[0].shape)
        total_feat_out.append(output)  # 然后分别存入全局 list 中
        total_feat_in.append(input)

    modules = model.named_modules()  #
    i = 0
    names = []
    for name, module in modules:
        i += 1
        names.append(name)
        module.register_forward_hook(hook_fn_forward)

    model = model.cuda()

    x = torch.randn((1, 3, 224, 224)).cuda()
    y = model(x)
    input_shapes = []
    print('==========Saved inputs and outputs==========')
    for idx in range(len(total_feat_in)):
        input_shapes.append(total_feat_in[idx][0].shape)
        # print('input: ', total_feat_in[idx][0].shape)
        # print('output: ', total_feat_out[idx][0].shape)
    memorys = 0
    print('memorys', memorys)
    for input_shape in input_shapes:
        memory = 1
        for i in input_shape:
            memory *= i
        # print(memory)
        memorys += memory

    print(memorys * 4 / 1000 / 1000)