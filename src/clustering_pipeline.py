import scanpy as sc
from sklearn.manifold import TSNE, trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from src.functions import pcs_from_knee
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from itertools import product
from kneed import KneeLocator
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')


class SingleCellPipeline:
    def __init__(self, adata, tsne_params, pca_params, umap_params, clustering_method, clustering_params):
        """
        Initialize the SingleCellPipeline instance.

        Parameters:
        - adata: Anndata object containing single-cell data.
        - tsne_params: Dictionary of parameters for t-SNE dimensionality reduction.
        - pca_params: Dictionary of parameters for PCA dimensionality reduction.
        - umap_params: Dictionary of parameters for UMAP dimensionality reduction.
        - clustering_method: String specifying the clustering method ('GMM' or 'KMeans').
        - clustering_params: Dictionary of parameters for the clustering method.
        """
        self.adata = adata
        self.tsne_params = tsne_params
        self.pca_params = pca_params
        self.umap_params = umap_params
        self.scaler = StandardScaler()
        self.clustering_method = clustering_method
        self.clustering_params = clustering_params
        self.results = {
            't-SNE': [],
            'UMAP': []
        }
        self.best_params = {
            't-SNE': [],
            'UMAP': [],
            'PCA': []
        }
        self.gmm_scores = {
            't-SNE': [],
            'UMAP': [],
            'PCA': []
        }
        self.kmeans_scores = {
            't-SNE': [],
            'UMAP': [],
            'PCA': []
        }

    # Dimensionality reduction
    def compute_tsne(self, perplexity, learning_rate, n_iter, n_components=2):
        """
        Perform t-SNE dimensionality reduction on the data.

        Parameters:
        - perplexity: Perplexity parameter for t-SNE.
        - learning_rate: Learning rate for t-SNE.
        - n_iter: Number of iterations for t-SNE.
        - n_components: Number of components for the reduced data (default is 2).

        Returns:
        - tsne_model: Reduced t-SNE model.
        """
        data_scaled = self.scaler.fit_transform(self.adata.X)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        tsne_model = tsne.fit_transform(data_scaled)
        return tsne_model

    def compute_pca(self, n_components):
        """
        Perform PCA dimensionality reduction on the data.

        Parameters:
        - n_components: Number of principal components to retain.

        Returns:
        - pca_model: PCA model object.
        """
        data_scaled = self.scaler.fit_transform(self.adata.X)
        pca = PCA(n_components=n_components)
        pca_model = pca.fit(data_scaled)
        return pca_model
    
    def compute_umap(self, n_neighbors, min_dist, n_components_umap):
        """
        Perform UMAP dimensionality reduction on the data.

        Parameters:
        - n_neighbors: Number of neighbors for UMAP.
        - min_dist: Minimum distance parameter for UMAP.
        - n_components_umap: Number of components for the reduced data.

        Returns:
        - umap_embedding: Reduced UMAP embeddings.
        """
        # Ensure parameters are integers
        n_neighbors = int(n_neighbors)
        min_dist = float(min_dist)
        n_components_umap = int(n_components_umap)
        data_scaled = self.scaler.fit_transform(self.adata.X)
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components_umap, random_state=42)
        umap_embedding = umap_model.fit_transform(data_scaled)
        return umap_embedding


    def grid_search(self):
        """
        Perform grid search to find optimal parameters for t-SNE and UMAP.

        Updates:
        - self.results: Stores results including trustworthiness scores.
        """
        # t-SNE grid search
        print('Performing grid search for t-SNE.')
        for n_component, perplexity, learning_rate, n_iter in tqdm(list(product(self.tsne_params['n_components'], self.tsne_params['perplexities'], self.tsne_params['learning_rates'], self.tsne_params['iterations'])), desc="t-SNE Grid Search"):
            X_tsne = self.compute_tsne(perplexity, learning_rate, n_iter)
            trustworthiness_score = trustworthiness(self.adata.X, X_tsne)
            self.results['t-SNE'].append({
                'n_components': n_component,
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'n_iter': n_iter,
                'trustworthiness': trustworthiness_score
            })

        # UMAP grid search
        print('Performing grid search for UMAP.')
        for params_tuple in tqdm(list(product(self.umap_params['n_neighbors'], self.umap_params['min_dist'], self.umap_params['n_components'])), desc="UMAP Grid Search"):
            n_neighbors, min_dist, n_components = params_tuple
            X_umap = self.compute_umap(n_neighbors, min_dist, n_components)
            trustworthiness_score = trustworthiness(self.adata.X, X_umap)
            self.results['UMAP'].append({
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'n_components': n_components,
                'trustworthiness': trustworthiness_score
            })

    # Find best parameters for each dimensionality reduction method
    def find_best_parameters(self, pca_params):
        """
        Determine the best parameters for t-SNE, UMAP, and PCA based on grid search results.

        Returns:
        - best_params: Dictionary containing the best parameters for each dimensionality reduction method.
        """
        # Compute median trustworthiness for t-SNE
        tsne_trustworthiness_scores = [result['trustworthiness'] for result in self.results['t-SNE']]
        median_tsne_index = np.argsort(tsne_trustworthiness_scores)[len(tsne_trustworthiness_scores)//2]
        median_tsne_parameters = self.results['t-SNE'][median_tsne_index].copy()

        # Compute median trustworthiness for UMAP
        umap_trustworthiness_scores = [result['trustworthiness'] for result in self.results['UMAP']]
        median_umap_index = np.argsort(umap_trustworthiness_scores)[len(umap_trustworthiness_scores)//2]
        median_umap_parameters = self.results['UMAP'][median_umap_index].copy()

        # Print messages and store results in best_params dictionary
        self.best_params['t-SNE'] = {
            **median_tsne_parameters
        }
        print(f"Best parameters for t-SNE: {self.best_params['t-SNE']} with trustworthiness score: {median_tsne_parameters['trustworthiness']}")
        del median_tsne_parameters['trustworthiness']  # Remove trustworthiness

        self.best_params['UMAP'] = {
            **median_umap_parameters
        }
        print(f"Best parameters for UMAP: {self.best_params['UMAP']} with trustworthiness score: {median_umap_parameters['trustworthiness']}")
        del median_umap_parameters['trustworthiness']  # Remove trustworthiness

        # Determine optimal PCA components
        pca_model = self.compute_pca(self.pca_params['n_components'][0])
        print('Plotting scree plot.')
        scree_plot(pca_model)
        optimal_pca_components = pcs_from_knee(pca_model)
        self.best_params['PCA'] = {
            'n_components': optimal_pca_components
        }
        
        return self.best_params
    
    # Clustering
    def clustering(self, clustering_method, clustering_params, best_params):
        """
        Perform clustering using specified method (GMM or KMeans) on selected embeddings.

        Parameters:
        - clustering_method: String specifying the clustering method ('GMM' or 'KMeans').
        - clustering_params: Dictionary of parameters for the clustering method.
        - best_params: Dictionary containing the best parameters for dimensionality reduction methods.

        Returns:
        - best_models: Dictionary containing the best clustering models for each method.
        """
        best_models = {}

        for key, params in best_params.items():
            if key == 'PCA':
                embeddings = self.compute_pca(n_components=params['n_components']).transform(self.adata.X)
            elif key == 't-SNE':
                embeddings = self.compute_tsne(perplexity=params['perplexity'],
                                               learning_rate=params['learning_rate'],
                                               n_iter=params['n_iter'],
                                               n_components=params['n_components'])
            elif key == 'UMAP':
                embeddings = self.compute_umap(n_neighbors=params['n_neighbors'],
                                               min_dist=params['min_dist'],
                                               n_components_umap=params['n_components'])
            else:
                print(f'Error: Parameters for {key} dimensionality reduction method were not found.')
                continue

            if clustering_method == 'GMM':
                best_bic = np.inf
                best_gmm_model = None
                best_gmm_params = None

                param_space = []  # To store all parameter combinations tried

                print(f"Performing GMM clustering for {key} embeddings...")
                for n_components in tqdm(clustering_params['n_components']):
                    for covariance_type in clustering_params['covariance_type']:
                        try:
                            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
                            gmm_labels = gmm.fit_predict(embeddings)
                            bic = gmm.bic(embeddings)

                            param_space.append({'n_components': n_components, 'covariance_type': covariance_type, 'bic_score': bic})

                            if bic < best_bic:
                                best_bic = bic
                                best_gmm_model = gmm
                                best_gmm_params = {'n_components': n_components, 'covariance_type': covariance_type}
                        except ValueError as e:
                            print(f"Skipping parameters {params} due to error: {e}")



                if best_gmm_model is not None:
                    best_models[key] = {
                        'labels': best_gmm_model.fit_predict(embeddings),
                        'bic_score': best_bic,
                        'posterior_probs': best_gmm_model.predict_proba(embeddings),
                        'gmm_params': best_gmm_params,
                        'parameter_space': param_space  # Store parameter space used
                    }
            elif clustering_method == 'KMeans':
                best_silhouette = -1
                best_km_model = None
                best_km_params = None

                param_space = []  # To store all parameter combinations tried

                print(f"Performing KMeans clustering for {key} embeddings...")
                for n_clusters in tqdm(clustering_params['n_clusters']):
                    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=clustering_params['n_init'], random_state=42)
                    km_labels = km.fit_predict(embeddings)
                    silhouette = silhouette_score(embeddings, km_labels)

                    param_space.append({'n_clusters': n_clusters, 'silhouette_score': silhouette})

                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_km_model = km
                        best_km_params = {'n_clusters': n_clusters}

                if best_km_model is not None:
                    best_models[key] = {
                        'labels': best_km_model.fit_predict(embeddings),
                        'silhouette_score': best_silhouette,
                        'km_params': best_km_params,
                        'parameter_space': param_space  # Store parameter space used
                    }
            else:
                print(f'Clustering method "{clustering_method}" not found. Try "GMM" or "KMeans".')

        return best_models
    
    # Visualize best embeddings
    def visualize_embeddings(self, best_params, best_models=None):
        for method, params in best_params.items():
            embeddings = None
            labels = None

            if method == 'PCA':
                pca_components = min(params['n_components'], 3)
                embeddings = self.compute_pca(pca_components).transform(self.adata.X)
                if best_models and method in best_models:
                    labels = best_models[method].get('labels', None)
            elif method == 't-SNE':
                embeddings = self.compute_tsne(perplexity=params['perplexity'], learning_rate=params['learning_rate'],
                                            n_iter=params['n_iter'], n_components=params['n_components'])
                if best_models and method in best_models:
                    labels = best_models[method].get('labels', None)
            elif method == 'UMAP':
                embeddings = self.compute_umap(n_neighbors=params['n_neighbors'], min_dist=params['min_dist'],
                                            n_components_umap=params['n_components'])
                if best_models and method in best_models:
                    labels = best_models[method].get('labels', None)

            if embeddings is not None:
                if embeddings.shape[1] == 3:
                    fig = px.scatter_3d(
                        x=embeddings[:, 0],
                        y=embeddings[:, 1],
                        z=embeddings[:, 2],
                        color=labels.astype(str) if labels is not None else None,
                        title=f'{method} 3D Visualization' + (' with Clustering' if labels is not None else ''),
                        labels={'color': 'Cluster'} if labels is not None else {}
                    )
                elif embeddings.shape[1] == 2:
                    fig = px.scatter(
                        x=embeddings[:, 0],
                        y=embeddings[:, 1],
                        color=labels.astype(str) if labels is not None else None,
                        title=f'{method} 2D Visualization' + (' with Clustering' if labels is not None else ''),
                        labels={'color': 'Cluster'} if labels is not None else {}
                    )
                else:
                    print(f"Embeddings for {method} have unexpected shape: {embeddings.shape}")
                    continue

                fig.update_layout(
                    title_x=0.5,
                    height=550,
                    width=650,
                    template='simple_white',
                    showlegend=True,
                    font=dict(family="Arial")
                )

                fig.show()


    def plot_posterior_distributions(self, best_models):
        for method, model_info in best_models.items():
            if 'labels' not in model_info or 'posterior_probs' not in model_info:
                continue

            labels = model_info['labels']
            posterior_probs = model_info['posterior_probs']
            n_components_gmm = model_info['gmm_params']['n_components']
            
            # Determine the unique clusters from labels
            clusters = np.unique(labels)

            fig = go.Figure()

            # Compute cumulative probabilities for each cluster
            for cluster in clusters:
                cluster_probs = posterior_probs[:, cluster]
                # Normalize probabilities to sum to 1
                cluster_probs /= np.sum(cluster_probs)
                # Calculate cumulative probabilities
                cumulative_probs = np.cumsum(cluster_probs)

                # Add bars with opaque 50%
                fig.add_trace(go.Bar(
                    x=np.arange(len(labels)),
                    y=cumulative_probs,
                    name=f'Cluster {cluster}',
                    marker_color=f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 0.5)',
                    hoverinfo='y+name'  # Show cumulative probability on hover
                ))

                # Add line connecting tops of bars
                if cluster < clusters[-1]:  # Only connect if not the last cluster
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(labels)),
                        y=cumulative_probs,
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ))

            fig.update_layout(
                title=f'Posterior Distributions per Cluster ({method})',
                title_x=0.5,
                xaxis_title='Cells',
                yaxis_title='Cumulative Probability',
                height=600,
                width=900,
                template='plotly_white',
                font=dict(family="Arial"),
                showlegend=True
            )

            fig.show()
            

    def save_labels(self, best_models, key='UMAP'):
        if key in best_models:
            model_info = best_models[key]

            if 'labels' in model_info:
                labels = model_info['labels']
            else:
                print(f"Error: No labels found for {key}")
                return

            # Create a DataFrame for the labels
            labels_df = pd.DataFrame({
                'labels': labels
            }, index=self.adata.obs_names)

            # Save the DataFrame to a CSV file named 'labels.csv'
            labels_df.to_csv('labels.csv', index=True)
            print("Labels saved to labels.csv")
        else:
            print(f"Error: {key} not found in best_models")


def pcs_from_knee(pca_model):
    """
    Calculate and print the number of principal components retained based on explained variance ratio.

    Parameters:
    - pca_data: PCA object from scikit-learn.

    Returns:
    - num_components: Number of principal components retained.
    """
    # Calculate explained variance ratio
    explained_variance_ratio = pca_model.explained_variance_ratio_

    # Generate x values (number of components)
    x = range(1, len(explained_variance_ratio) + 1)

    # Use KneeLocator to find the knee point
    kn = KneeLocator(x, explained_variance_ratio, curve='convex', direction='decreasing')

    # Number of principal components retained
    num_components_retained = kn.knee

    # Print message
    print(f"Number of principal components retained: {num_components_retained}")

    return num_components_retained

def scree_plot(pca):
    explained_variance = pca.explained_variance_ratio_
    trace = go.Scatter(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        mode='lines+markers',
        marker=dict(color='#de8b3e'),
        line=dict(width=2)
    )

    layout = go.Layout(
        title='Scree Plot',
        title_x=0.5,
        xaxis=dict(title='Principal Component'),
        yaxis=dict(title='Explained Variance Ratio'),
        template='simple_white',
        font=dict(family="Arial", size=15)
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(height=450, width=550)
    fig.show()
