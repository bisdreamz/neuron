package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Debug the specific failing language model test to understand why it's not learning the expected pattern.
 */
public class SimpleLanguageModelDebugTest {
    
    @Test
    public void debugSimpleRepetitivePattern() {
        // Create the exact same model as the failing test
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0001f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8)) // Small vocab, small embedding
                .layer(Layers.hiddenGruLastNormalized(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Create the exact same dataset as the failing test
        List<String> pattern = Arrays.asList("a", "b", "c");
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Generate 100 sequences of length 3
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < pattern.size(); j++) {
                String[] seq = new String[3];
                for (int k = 0; k < 3; k++) {
                    seq[k] = pattern.get((i * 3 + j + k) % pattern.size());
                }
                sequences.add(seq);
                targets.add(pattern.get((i * 3 + j + 3) % pattern.size()));
            }
        }
        
        // Print a few examples to understand the pattern
        System.out.println("Training data examples:");
        for (int i = 0; i < Math.min(10, sequences.size()); i++) {
            System.out.println(Arrays.toString(sequences.get(i)) + " -> " + targets.get(i));
        }
        
        // Train with the exact same configuration
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(10)
            .shuffle(false) // Keep pattern order
            .verbosity(0)
            .build();
            
        model.trainBulk(sequences, targets, config);
        
        // Test predictions on specific sequences
        
        // After seeing "a b c", should predict "a" with very high confidence
        String[] testSeq1 = {"a", "b", "c"};
        String pred1 = model.predictNext(testSeq1);
        float[] probs1 = model.predictProbabilities(testSeq1);
        System.out.println("\nAfter 'a b c': predicted='" + pred1 + "', expected='a'");
        System.out.println("Probability distribution: " + Arrays.toString(probs1));
        
        // After seeing "b c a", should predict "b"
        String[] testSeq2 = {"b", "c", "a"};
        String pred2 = model.predictNext(testSeq2);
        float[] probs2 = model.predictProbabilities(testSeq2);
        System.out.println("\nAfter 'b c a': predicted='" + pred2 + "', expected='b'");
        System.out.println("Probability distribution: " + Arrays.toString(probs2));
        
        // After seeing "c a b", should predict "c"
        String[] testSeq3 = {"c", "a", "b"};
        String pred3 = model.predictNext(testSeq3);
        float[] probs3 = model.predictProbabilities(testSeq3);
        System.out.println("\nAfter 'c a b': predicted='" + pred3 + "', expected='c'");
        System.out.println("Probability distribution: " + Arrays.toString(probs3));
        
        // Check vocabulary
        System.out.println("\nVocabulary size: " + model.getVocabularySize());
        for (int i = 0; i < model.getVocabularySize(); i++) {
            InputSequenceEmbeddingLayer embedLayer = (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
            String word = embedLayer.getWord(i);
            System.out.println("Token " + i + ": '" + word + "'");
        }
        
        // Let's also check if the pattern makes sense
        System.out.println("\nPattern analysis:");
        Map<String, Integer> sequenceCounts = new HashMap<>();
        Map<String, Map<String, Integer>> transitions = new HashMap<>();
        
        for (int i = 0; i < sequences.size(); i++) {
            String seqKey = Arrays.toString(sequences.get(i));
            String target = targets.get(i);
            
            sequenceCounts.put(seqKey, sequenceCounts.getOrDefault(seqKey, 0) + 1);
            
            transitions.putIfAbsent(seqKey, new HashMap<>());
            Map<String, Integer> targetCounts = transitions.get(seqKey);
            targetCounts.put(target, targetCounts.getOrDefault(target, 0) + 1);
        }
        
        for (Map.Entry<String, Map<String, Integer>> entry : transitions.entrySet()) {
            System.out.println(entry.getKey() + " -> " + entry.getValue());
        }
    }
}