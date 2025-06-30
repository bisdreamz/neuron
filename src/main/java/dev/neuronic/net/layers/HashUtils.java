package dev.neuronic.net.layers;

import net.openhft.hashing.LongHashFunction;

/**
 * Utility class for high-quality hashing across the neural network library.
 * 
 * <p>Provides consistent, production-grade hash functions for both hashed embeddings 
 * and distributed dictionary indexing. Uses OpenHFT's XXH3 for optimal performance
 * and distribution quality.
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
    
    // Shared high-quality hash function across the library
    private static final LongHashFunction HASH_FUNCTION = LongHashFunction.xx3();
    
    /**
     * Hash a string to an integer for use with hashed embeddings.
     * 
     * <p>Uses production-grade XXH3 hash function for optimal distribution
     * across embedding buckets. No additional mixing needed.
     * 
     * @param s the string to hash (null returns 0)
     * @return hash code as integer
     */
    public static int hashString(String s) {
        if (s == null) return 0;
        
        // Use high-quality XXH3 hash function
        long hash = HASH_FUNCTION.hashChars(s);
        return (int) hash; // Truncate to 32 bits
    }
    
    /**
     * Generate distributed hash for dictionary indexing.
     * Provides consistent hashing across Dictionary and LRUDictionary.
     * 
     * @param value the value to hash
     * @return distributed hash code
     */
    public static long distributedHash(Object value) {
        if (value == null) return 0;
        return HASH_FUNCTION.hashChars(value.toString());
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
        
        // Combine strings with separator for consistent hashing
        StringBuilder combined = new StringBuilder();
        for (int i = 0; i < strings.length; i++) {
            if (i > 0) combined.append('\0'); // Null separator
            combined.append(strings[i]);
        }
        
        // Use consistent XXH3 hash
        long hash = HASH_FUNCTION.hashChars(combined);
        return (int) hash;
    }
}