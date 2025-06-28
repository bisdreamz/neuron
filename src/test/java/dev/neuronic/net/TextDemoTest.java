package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetString;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to demonstrate the simplified WikiText2Demo approach.
 */
public class TextDemoTest {
    
    @Test
    public void testSimpleLanguageModel() {
        // Create a small language model
        int sequenceLength = 5;
        int vocabSize = 100;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        NeuralNet net = NeuralNet.newBuilder()
                .input(sequenceLength)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
                .layer(Layers.inputSequenceEmbedding(sequenceLength, vocabSize, embeddingDim))
                .layer(Layers.hiddenGruLast(hiddenSize))
                .output(Layers.outputSoftmaxCrossEntropy(vocabSize));
        
        SimpleNetString model = SimpleNet.ofStringClassification(net);
        
        // Prepare some simple training data
        List<Object> sequences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        
        // Create simple sequences
        sequences.add(new String[]{"the", "cat", "sat", "on", "the"});
        labels.add("mat");
        
        sequences.add(new String[]{"the", "dog", "ran", "in", "the"});
        labels.add("park");
        
        sequences.add(new String[]{"a", "bird", "flew", "over", "the"});
        labels.add("tree");
        
        sequences.add(new String[]{"the", "sun", "shone", "on", "the"});
        labels.add("water");
        
        // Train
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
                .batchSize(2)
                .epochs(5)
                .verbosity(0) // Silent for test
                .build();
        
        SimpleNetTrainingResult result = model.trainBulk(sequences.toArray(new Object[0]), labels.toArray(new String[0]), config);
        assertNotNull(result);
        assertTrue(result.getEpochsTrained() > 0);
        
        // Test prediction - should return a string
        String[] testSequence = {"the", "cat", "sat", "on", "the"};
        String prediction = model.predictString(testSequence);
        assertNotNull(prediction);
        
        // Verify vocabulary was built
        InputSequenceEmbeddingLayer embeddingLayer = 
            (InputSequenceEmbeddingLayer) model.getNetwork().getInputLayer();
        assertTrue(embeddingLayer.hasWord("the"));
        assertTrue(embeddingLayer.hasWord("cat"));
        
        // Test unknown words
        String[] unknownSequence = {"unknown", "words", "not", "in", "vocab"};
        String unknownPrediction = model.predictString(unknownSequence);
        assertNotNull(unknownPrediction); // Should handle gracefully with <unk>
        
        System.out.println("Language model test passed!");
        System.out.println("Vocabulary size: " + embeddingLayer.getVocabularySize());
        System.out.println("Sample prediction: " + prediction);
    }
}