package dev.neuronic.net.layers;

/**
 * Utility class for hashing strings to integers for use with HASHED_EMBEDDING features.
 * 
 * <p>When using hashed embeddings, you need to convert your string values to hash codes
 * before passing them to the neural network. This utility provides consistent hashing.
 * 
 * <p><b>Example usage:</b>
 * <pre>{@code
 * // Configure model with hashed embedding
 * Feature.hashedEmbedding(10_000, 16, "domain")
 * 
 * // Hash strings before feeding to model
 * String domain = "example.com";
 * float domainHash = (float) HashUtils.hashString(domain);
 * model.predict(new float[] { domainHash, ... });
 * }</pre>
 */
public final class HashUtils {
    
    private HashUtils() {} // Utility class
    
    /**
     * Hash a string to an integer for use with hashed embeddings.
     * 
     * <p>Uses a high-quality hash function with additional mixing for better
     * distribution across embedding buckets.
     * 
     * @param s the string to hash (null returns 0)
     * @return hash code as integer
     */
    public static int hashString(String s) {
        if (s == null) return 0;
        
        // Use Java's string hash as base
        int h = s.hashCode();
        
        // Additional mixing for better distribution
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        
        return h;
    }
    
    /**
     * Hash multiple strings together (useful for composite keys).
     * 
     * <p>Example: hashStrings("user123", "domain.com") for user-domain pairs
     * 
     * @param strings strings to combine and hash
     * @return combined hash code
     */
    public static int hashStrings(String... strings) {
        if (strings == null || strings.length == 0) return 0;
        
        int result = 1;
        for (String s : strings) {
            int h = hashString(s);
            result = 31 * result + h;
        }
        
        // Final mixing
        result ^= result >>> 16;
        result *= 0x85ebca6b;
        result ^= result >>> 13;
        
        return result;
    }
}