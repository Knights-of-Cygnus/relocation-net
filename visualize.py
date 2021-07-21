from torch.utils import data
from pathlib import Path
from data.sevenscenes import Scenes, Modes, SevenScenes

def main():
    from data.visutil import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    import torchvision.transforms as transforms
    from PIL.Image import Image
    seq: Scenes = 'chess'
    mode: Modes = 2
    num_workers = 1
    # num_workers = 6
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        # FIXME: Numpy complains __array__ takes 1 arg but 2 were given, fix.
        # Image.__array__,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    p = Path('..') / '..' / 'Datasets' / '7scenes'
    dset = SevenScenes(seq, p, True, transform,
                       mode=mode)
    print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
                                                               len(dset)))

    data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
                                  num_workers=num_workers)

    batch_count = 0
    N = 2
    for batch in data_loader:
        print('Minibatch {:d}'.format(batch_count))
        if mode < 2:
            show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        elif mode == 2:
            lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
            rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
            show_stereo_batch(lb, rb)

        batch_count += 1
        if batch_count >= N:
            break

if __name__ == '__main__':
    main()
