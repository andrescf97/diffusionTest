import argparse
import sys

def main():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    g = sub.add_parser('train')
    g.add_argument('--config', default='configs/default.yaml')

    s = sub.add_parser('sample')
    s.add_argument('--ckpt', required=True)
    s.add_argument('--out', default='samples')

    gen = sub.add_parser('generate')
    gen.add_argument('--out', default='data/synthetic.npy')

    args = p.parse_args()
    if args.cmd == 'train':
        from scripts.train import train
        train(args.config)
    elif args.cmd == 'sample':
        from scripts.sample import sample
        sample(args.ckpt, args.out)
    elif args.cmd == 'generate':
        from scripts.generate_synthetic import make_synthetic
        make_synthetic(args.out)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
