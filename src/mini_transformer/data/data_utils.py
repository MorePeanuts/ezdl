import matplotlib.pyplot as plt


def show_images(imgs, n_rows, n_cols, titles=None, scale=1.5):
    figsize = (n_cols * scale, n_rows * scale)
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # type: ignore
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().numpy()
        except Exception:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def format_instruction(
    entry: dict,
    instruction_template: str | None = None,
    response_template: str | None = None,
    instruction_key='instruction',
    input_key='input',
    output_key='output',
):
    if instruction_template is None:
        instruction_template = (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.'
            '\n\n### Instruction:\n{instruction}'
            '\n\n### Input:\n{input}'
        )
    if response_template is None:
        response_template = '\n\n### Response:\n{output}'

    instruction_text = instruction_template.format(
        instruction=entry.get(instruction_key, ''),
        input=entry.get(input_key, ''),
    )
    response_text = response_template.format(output=entry.get(output_key, ''))

    return instruction_text, response_text
