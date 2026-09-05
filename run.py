# run.py
import argparse
import jax.random as jrd
import src.config as cfg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous-Time Flow Matching CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Expanded Execution Modes
    parser.add_argument("--mode", choices=["train", "eval", "heatmap", "vector", "all"], default="train", help="Select the pipeline stage to execute. Use 'eval' for high-D datasets.")
    parser.add_argument("--interactive", action="store_true", help="Render matplotlib plots in GUI windows instead of saving them to disk.")
    
    # 2. Target & Dataset Controls
    parser.add_argument("--resolution", default=cfg.HEATMAP_RESOLUTION, help="Resolution of the heatmap generated in heatmap mode") 
    parser.add_argument("--shape", choices=["square", "spiral", "ring", "personalized"], default=cfg.SHAPE)
    parser.add_argument("--train-data", type=str, default="data/train.npy", help="Path to binary train data (Used if shape='personalized').")
    parser.add_argument("--test-data", type=str, default="data/test.npy", help="Path to binary test data (Used if mode='eval').")
    
    # 3. Model & Training Hyperparameters
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--width", type=int, default=cfg.WIDTH)
    parser.add_argument("--depth", type=int, default=cfg.DEPTH)
    parser.add_argument("--epochs", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4096)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config memory BEFORE any deep imports
    cfg.HEATMAP_RESOLUTION=args.resolution
    cfg.SHAPE = args.shape
    cfg.WIDTH = args.width
    cfg.DEPTH = args.depth
    if args.seed != cfg.SEED:
        cfg.SEED = args.seed
        cfg.KEY = jrd.key(args.seed)

    # Late imports
    import src.targets as tg
    
    vram_data = None
    
    # Dataset Loading and Dynamic Dimension Adjustment
    if cfg.SHAPE == "personalized":
        if args.mode in ["train", "all"]:
            print(f"Locking {args.train_data} into GPU VRAM...")
            vram_data, true_dim = tg.load_dataset_to_vram(args.train_data)
        elif args.mode == "eval":
            print(f"Locking {args.test_data} into GPU VRAM...")
            vram_data, true_dim = tg.load_dataset_to_vram(args.test_data)
            
        # Overwrite IN_SIZE dynamically so the model matches the dataset dimensions
        cfg.IN_SIZE = true_dim
        cfg.OUT_SIZE = true_dim
        print(f"Architecture dynamically adapted to {true_dim}D input.")

    # Execution Routing
    if args.mode in ["train", "all"]:
        import train as tr
        print(f"Training | Target: {cfg.SHAPE} | Epochs: {args.epochs} | Batch: {args.batch_size}")
        tr.train(epochs=args.epochs, batch_size=args.batch_size, vram_array=vram_data)

    if args.mode == "eval":
        # Create an evaluation script to batch over vram_data and compute test-set NLL
        import src.eval as ev
        print(f"Evaluating Test Set NLL | Target: {cfg.SHAPE}")
        ev.evaluate_nll(batch_size=args.batch_size, vram_array=vram_data)

    if args.mode in ["heatmap", "vector", "all"]:
        if cfg.IN_SIZE > 2:
            print("Warning: Visualizations are disabled for high-dimensional datasets (>2D).")
        else:
            import src.viz as viz
            if args.mode in ["heatmap", "all"]:
                viz.plot_heatmap(args.interactive)
            if args.mode in ["vector", "all"]:
                viz.plot_vector_field(args.interactive)

if __name__ == "__main__":
    main()
