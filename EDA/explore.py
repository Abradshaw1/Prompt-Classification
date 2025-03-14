import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Load the CSV file
df = pd.read_csv('df_with_validation_embeddings_MPnet.csv')

df['Embeddings'] = df['Embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Apply preprocessing to the 'Embeddings' column
embeddings = np.stack(df['Embeddings'].values)

# Apply UMAP
reducer = umap.UMAP()
umap_embeddings = reducer.fit_transform(embeddings)

# Add the UMAP embeddings to the DataFrame
df['UMAP_1'] = umap_embeddings[:, 0]
df['UMAP_2'] = umap_embeddings[:, 1]

sns.set(style="whitegrid")

# Create the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP_1', 
    y='UMAP_2', 
    hue='Department',  # Color by the 'Malicious' column
    palette='Set1',      # Use a color palette
    data=df, 
    s=20,                  # Size of the points
    alpha = 0.5
)

# Add titles and labels
plt.title('UMAP Visualization of MPnet Embeddings', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position

plt.tight_layout()
plt.show()
