package dev.neuronic.net.layers;

/**
 * Feature type for high-cardinality categorical data using hash-based embeddings.
 * 
 * <p><b>What:</b> Maps arbitrary strings to embeddings without vocabulary limits.
 * Uses multiple hash functions to reduce collision impact while maintaining
 * a fixed memory footprint.
 * 
 * <p><b>When to use:</b>
 * <ul>
 *   <li>High-cardinality features (domains, user IDs, app bundles)</li>
 *   <li>Online learning where vocabulary isn't known upfront</li>
 *   <li>When you need to handle millions of unique values efficiently</li>
 * </ul>
 * 
 * <p><b>Why use this:</b>
 * <ul>
 *   <li>No vocabulary size limits - handles any input</li>
 *   <li>Fixed memory usage regardless of unique values seen</li>
 *   <li>Collision-resistant through multi-hash averaging</li>
 *   <li>No dictionary management overhead</li>
 * </ul>
 * 
 * <p><b>Example usage:</b>
 * <pre>{@code
 * // For domains with potentially millions of unique values
 * Feature.hashedEmbedding(10_000, 16, "domain")
 * 
 * // For app bundles with unknown vocabulary
 * Feature.hashedEmbedding(50_000, 32, "app_bundle")
 * }</pre>
 * 
 * <p><b>How it works:</b>
 * Instead of building a vocabulary and assigning indices, this feature:
 * <ol>
 *   <li>Hashes the input string using 3 different hash functions</li>
 *   <li>Maps each hash to a position in the embedding table (modulo hash buckets)</li>
 *   <li>Averages the embeddings from all 3 positions</li>
 * </ol>
 * 
 * This approach trades perfect disambiguation for unlimited scale and zero
 * vocabulary management overhead.
 */
public final class HashedEmbeddingFeature {
    
    private final int hashBuckets;
    private final int embeddingDim;
    private final String name;
    private final int numHashes;
    
    /**
     * Creates a hashed embedding feature.
     * 
     * @param hashBuckets number of hash buckets (embedding table rows)
     * @param embeddingDim dimension of each embedding vector
     * @param name feature name for debugging
     * @param numHashes number of hash functions to use (typically 3)
     */
    public HashedEmbeddingFeature(int hashBuckets, int embeddingDim, String name, int numHashes) {
        this.hashBuckets = hashBuckets;
        this.embeddingDim = embeddingDim;
        this.name = name;
        this.numHashes = numHashes;
    }
    
    public int getOutputSize() {
        return embeddingDim;
    }
    
    public Feature.Type getType() {
        return Feature.Type.HASHED_EMBEDDING;
    }
    
    /**
     * Gets the number of hash buckets (embedding table size).
     */
    public int getHashBuckets() {
        return hashBuckets;
    }
    
    /**
     * Gets the embedding dimension.
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }
    
    /**
     * Gets the number of hash functions used.
     */
    public int getNumHashes() {
        return numHashes;
    }
    
    /**
     * Gets the feature name.
     */
    public String getName() {
        return name;
    }
    
    @Override
    public String toString() {
        return String.format("HashedEmbedding(%d buckets, %d dim, %d hashes, \"%s\")", 
            hashBuckets, embeddingDim, numHashes, name);
    }
}