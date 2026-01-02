import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='CSIH Model Arguments')
    
    # Training Hyperparameters
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size (reduced for subgraph processing)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--latdim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dataset_path', type=str, default='./interactions.csv', help='Path to dataset')
    
    # CSIH Specific Arguments
    parser.add_argument('--n_intents', type=int, default=8, help='Number of latent intents (K)')
    parser.add_argument('--subgraph_radius', type=int, default=2, help='Radius for local subgraph extraction (k-hop)')
    parser.add_argument('--max_dist_label', type=int, default=10, help='Maximum distance label for two-anchor encoding')
    
    # Loss Weights
    parser.add_argument('--lambda_gl', type=float, default=0.1, help='Weight for Global-Local Contrastive Loss')
    parser.add_argument('--lambda_intent', type=float, default=0.01, help='Weight for Intent Regularization')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L2 Regularization weight')
    
    # Temperature for Contrastive Loss
    parser.add_argument('--temp_gl', type=float, default=0.2, help='Temperature for GL contrastive loss')
    
    return parser.parse_args()

args = ParseArgs()
