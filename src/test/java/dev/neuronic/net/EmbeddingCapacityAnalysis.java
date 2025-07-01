package dev.neuronic.net;

import org.junit.jupiter.api.Test;

/**
 * Analyze embedding capacity for production scale:
 * - 10,000+ unique zone IDs
 * - 5,000+ unique app bundles
 * - 100s of publisher IDs
 */
public class EmbeddingCapacityAnalysis {
    
    @Test
    public void analyzeEmbeddingCapacity() {
        System.out.println("=== EMBEDDING CAPACITY ANALYSIS ===\n");
        
        // Current configuration
        System.out.println("CURRENT CONFIGURATION:");
        System.out.println("- pubid: 100 IDs × 8 dims = 800 parameters");
        System.out.println("- app_bundle: 10,000 buckets × 16 dims = 160,000 parameters (hashed)");
        System.out.println("- zone_id: 4,000 IDs × 12 dims = 48,000 parameters");
        System.out.println("Total embedding parameters: ~208,800\n");
        
        // Actual scale
        System.out.println("YOUR PRODUCTION SCALE:");
        System.out.println("- 10,000+ unique zone IDs");
        System.out.println("- 5,000+ unique app bundles");
        System.out.println("- 100s of publishers");
        System.out.println("- Plus OS, device type, connection type features\n");
        
        // Capacity analysis
        System.out.println("CAPACITY ISSUES:");
        System.out.println("1. Zone ID: 4,000 capacity < 10,000+ actual");
        System.out.println("   → 60% of zones share embeddings (collision)");
        System.out.println("2. App bundle: Hashed to 10k buckets OK, but low dims");
        System.out.println("   → 16 dims may be insufficient for 5k+ unique apps\n");
        
        // Recommended configuration
        System.out.println("RECOMMENDED CONFIGURATION:");
        System.out.println("```java");
        System.out.println("Feature[] features = {");
        System.out.println("    Feature.oneHot(10, \"os\"),");
        System.out.println("    Feature.embeddingLRU(200, 16, \"pubid\"),        // 200×16 = 3,200 params");
        System.out.println("    Feature.hashedEmbedding(50_000, 32, \"app_bundle\"), // 50k×32 = 1.6M params");
        System.out.println("    Feature.embeddingLRU(15_000, 24, \"zone_id\"),   // 15k×24 = 360k params");
        System.out.println("    Feature.oneHot(7, \"device_type\"),");
        System.out.println("    Feature.oneHot(5, \"connection_type\"),");
        System.out.println("    Feature.passthrough(\"bid_floor\")");
        System.out.println("};");
        System.out.println("```\n");
        
        System.out.println("BENEFITS:");
        System.out.println("- Zone ID: 15k capacity > 10k actual (50% headroom)");
        System.out.println("- App bundle: 50k buckets with 32 dims (better representation)");
        System.out.println("- Publisher: 200 capacity with 16 dims (richer features)");
        System.out.println("- Total: ~2M embedding parameters (10x current)\n");
        
        System.out.println("HIDDEN LAYER RECOMMENDATIONS:");
        System.out.println("With 2M+ embedding parameters, increase hidden layers:");
        System.out.println("```java");
        System.out.println(".layer(Layers.hiddenDenseRelu(512))  // Was 256");
        System.out.println(".layer(Layers.hiddenDenseRelu(256))  // Was 128");
        System.out.println(".layer(Layers.hiddenDenseRelu(128))  // Was 64");
        System.out.println(".layer(Layers.hiddenDenseRelu(64))   // Extra layer");
        System.out.println("```\n");
        
        System.out.println("MEMORY IMPACT:");
        long embeddingMemory = (200L * 16 + 50_000L * 32 + 15_000L * 24) * 4; // float = 4 bytes
        long hiddenMemory = (512 * 256 + 256 * 128 + 128 * 64 + 64 * 1) * 4;
        long totalMemory = embeddingMemory + hiddenMemory;
        
        System.out.printf("- Embedding memory: %.1f MB\n", embeddingMemory / 1024.0 / 1024.0);
        System.out.printf("- Hidden layer memory: %.1f MB\n", hiddenMemory / 1024.0 / 1024.0);
        System.out.printf("- Total model memory: %.1f MB\n", totalMemory / 1024.0 / 1024.0);
        System.out.println("\nThis is very reasonable for production use!");
    }
}