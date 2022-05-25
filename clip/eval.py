from tqdm import tqdm
from hfai.datasets import ImageNet

import torch
from torchvision import transforms
from data import (
    tokenize,
    imagenet_classnames,
    openai_imagenet_template,
)
import clip


# follow openclip
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    zeroshot_weights = []
    for classname in imagenet_classnames:
        texts = [template(classname) for template in openai_imagenet_template]
        texts = tokenize(texts).cuda()
        class_embeds = model.encode_text(texts)
        class_embeds /= class_embeds.norm(dim=-1, keepdim=True)
        class_embed = class_embeds.mean(dim=0)
        class_embed /= class_embed.norm()
        zeroshot_weights.append(class_embed)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    correct1, correct5, total = torch.zeros(3).cuda()
    for batch in tqdm(dataloader):
        images, labels = [x.cuda(non_blocking=True) for x in batch]
        img_embeds = model.encode_image(images)

        img_embeds /= img_embeds.norm(dim=-1, keepdim=True)
        logits = img_embeds @ zeroshot_weights

        _, preds = logits.topk(5, -1, True, True)
        correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
        correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
        total += images.size(0)

    acc1 = 100 * correct1.item() / total.item()
    acc5 = 100 * correct5.item() / total.item()
    print(f"Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%", flush=True)

    return correct1.item() / total.item()


def main():
    batch_size = 64
    model = "clip_rn50"
    ckpt_path = "output/RN50/best.pt"

    # model, dataloader, optimizer
    model = clip.__dict__[model]()
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model'])
    print(f"Loaded model from {ckpt_path}")

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(model.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = ImageNet(split="val", transform=transform, check_data=False)
    dataloader = dataset.loader(batch_size, num_workers=8, pin_memory=True)
    acc = validate(model, dataloader)


if __name__ == "__main__":
    main()
