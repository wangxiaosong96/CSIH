import torch
import torch.optim as optim
from Params import args
from DataHandler import DataHandler
from CSIH import CSIH

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load Data
# This will process 'interactions.csv'
handler = DataHandler(args)
loader = handler.get_loader('train')

# 3. Initialize CSIH Model
# User + Items total count handled by embeddings
model = CSIH(args, handler.num_users, handler.num_items).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda_reg)

# 4. Prepare Global Graph (Static)
g_edge_index, g_edge_attr = handler.get_global_graph_tensor()
g_edge_index = g_edge_index.to(device)
g_edge_attr = g_edge_attr.to(device)

# 5. Training Loop
print("Start Training CSIH...")

for epoch in range(args.epoch):
    total_loss = 0
    model.train()
    
    # Iterate over batches of subgraphs
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        # Pass the global graph structure AND the local batch of subgraphs
        pred, gl_pairs, intent_probs = model(g_edge_index, g_edge_attr, batch)
        
        # Calculate Loss
        loss = model.calc_loss(pred, batch.y, gl_pairs, intent_probs, args)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / len(loader)
    print(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} ===")

print("Training Finished.")

# Optional: Prediction Example
model.eval()
print("\n--- Example Prediction Output ---")
with torch.no_grad():
    # Take one batch to show predictions
    sample_batch = next(iter(loader)).to(device)
    preds, _, _ = model(g_edge_index, g_edge_attr, sample_batch)
    
    # Map back IDs to Names
    u_ids = sample_batch.root_u.cpu().numpy()
    i_ids = sample_batch.root_i.cpu().numpy()
    
    for k in range(min(5, len(u_ids))):
        gene_name = handler.user_encoder.inverse_transform([u_ids[k]])[0]
        drug_name = handler.item_encoder.inverse_transform([i_ids[k]])[0]
        print(f"Gene: {gene_name} <-> Drug: {drug_name} | Pred Score: {preds[k]:.4f}")