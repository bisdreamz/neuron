package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetString;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that ensure SimpleNet types cannot be loaded as different types.
 * 
 * Each SimpleNet type writes its own unique type ID during serialization.
 * When loading, if the type ID doesn't match the expected type, an IOException
 * is thrown with an error message indicating the mismatch.
 * 
 * Note: The error messages now contain user-friendly type names and
 * suggest the correct loading method to use. For example:
 * "Type mismatch: The file contains a SimpleNetString model, but you're
 * trying to load it as SimpleNetLanguageModel. Use SimpleNetString.load() instead."
 */
public class SimpleNetCrossTypeLoadingTest {

    @TempDir
    Path tempDir;

    @Test
    void testLanguageModelCannotBeLoadedAsString() throws IOException {
        // Create and save a SimpleNetLanguageModel
        Path modelPath = tempDir.resolve("language_model.bin");
        
        // Create a language model neural network
        NeuralNet languageNet = NeuralNet.newBuilder()
            .input(10) // sequence length
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputSequenceEmbedding(10, 100, 16)) // seq_len, vocab_size, embedding_dim
            .layer(Layers.hiddenGruLast(32))
            .output(Layers.outputSoftmaxCrossEntropy(100)); // vocab_size
            
        SimpleNetLanguageModel languageModel = SimpleNet.ofLanguageModel(languageNet);
        languageModel.save(modelPath);

        // Try to load it as SimpleNetString - should fail
        IOException exception = assertThrows(IOException.class, () -> {
            SimpleNetString.load(modelPath);
        });

        // Verify error message contains meaningful type information
        String message = exception.getMessage();
        assertTrue(message.contains("Type mismatch"), 
            "Error message should indicate type mismatch");
        assertTrue(message.contains("SimpleNetLanguageModel"), 
            "Error message should mention that file contains SimpleNetLanguageModel");
        assertTrue(message.contains("trying to load it as SimpleNetString"), 
            "Error message should mention attempted type");
        assertTrue(message.contains("Use SimpleNetLanguageModel.load() instead"), 
            "Error message should suggest correct loading method");
    }

    @Test
    void testStringCannotBeLoadedAsLanguageModel() throws IOException {
        // Create and save a SimpleNetString
        Path modelPath = tempDir.resolve("string_model.bin");
        
        // Create a string classification neural network
        NeuralNet stringNet = NeuralNet.newBuilder()
            .input(100) // features
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10)); // 10 classes
            
        SimpleNetString stringModel = SimpleNet.ofStringClassification(stringNet);
        stringModel.save(modelPath);

        // Try to load it as SimpleNetLanguageModel - should fail
        IOException exception = assertThrows(IOException.class, () -> {
            SimpleNetLanguageModel.load(modelPath);
        });

        // Verify error message contains meaningful type information
        String message = exception.getMessage();
        assertTrue(message.contains("Type mismatch"), 
            "Error message should indicate type mismatch");
        assertTrue(message.contains("SimpleNetString"), 
            "Error message should mention that file contains SimpleNetString");
        assertTrue(message.contains("trying to load it as SimpleNetLanguageModel"), 
            "Error message should mention attempted type");
        assertTrue(message.contains("Use SimpleNetString.load() instead"), 
            "Error message should suggest correct loading method");
    }

    @Test
    void testIntCannotBeLoadedAsFloat() throws IOException {
        // Create and save a SimpleNetInt
        Path modelPath = tempDir.resolve("int_model.bin");
        
        // Create an int classification neural network
        NeuralNet intNet = NeuralNet.newBuilder()
            .input(100) // features
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10)); // 10 classes
            
        SimpleNetInt intModel = SimpleNet.ofIntClassification(intNet);
        intModel.save(modelPath);

        // Try to load it as SimpleNetFloat - should fail
        IOException exception = assertThrows(IOException.class, () -> {
            SimpleNetFloat.load(modelPath);
        });

        // Verify error message contains meaningful type information
        String message = exception.getMessage();
        assertTrue(message.contains("Type mismatch"), 
            "Error message should indicate type mismatch");
        assertTrue(message.contains("SimpleNetInt"), 
            "Error message should mention that file contains SimpleNetInt");
        assertTrue(message.contains("trying to load it as SimpleNetFloat"), 
            "Error message should mention attempted type");
        assertTrue(message.contains("Use SimpleNetInt.load() instead"), 
            "Error message should suggest correct loading method");
    }

    @Test
    void testEachTypeCanLoadItsOwnSavedModel() throws IOException {
        // Verify that each type can still load its own saved models correctly
        
        // SimpleNetLanguageModel
        Path languageModelPath = tempDir.resolve("language_model_valid.bin");
        NeuralNet languageNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputSequenceEmbedding(10, 100, 16))
            .layer(Layers.hiddenGruLast(32))
            .output(Layers.outputSoftmaxCrossEntropy(100));
        SimpleNetLanguageModel languageModel = SimpleNet.ofLanguageModel(languageNet);
        languageModel.save(languageModelPath);
        SimpleNetLanguageModel loadedLanguageModel = SimpleNetLanguageModel.load(languageModelPath);
        assertNotNull(loadedLanguageModel);

        // SimpleNetString
        Path stringModelPath = tempDir.resolve("string_model_valid.bin");
        NeuralNet stringNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        SimpleNetString stringModel = SimpleNet.ofStringClassification(stringNet);
        stringModel.save(stringModelPath);
        SimpleNetString loadedStringModel = SimpleNetString.load(stringModelPath);
        assertNotNull(loadedStringModel);

        // SimpleNetInt
        Path intModelPath = tempDir.resolve("int_model_valid.bin");
        NeuralNet intNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        SimpleNetInt intModel = SimpleNet.ofIntClassification(intNet);
        intModel.save(intModelPath);
        SimpleNetInt loadedIntModel = SimpleNetInt.load(intModelPath);
        assertNotNull(loadedIntModel);

        // SimpleNetFloat
        Path floatModelPath = tempDir.resolve("float_model_valid.bin");
        NeuralNet floatNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputLinearRegression(1)); // Single regression output
        SimpleNetFloat floatModel = SimpleNet.ofFloatRegression(floatNet);
        floatModel.save(floatModelPath);
        SimpleNetFloat loadedFloatModel = SimpleNetFloat.load(floatModelPath);
        assertNotNull(loadedFloatModel);
    }

    @Test
    void testAllCrossTypeCombinationsFail() throws IOException {
        // Create one of each type
        Path languageModelPath = tempDir.resolve("lang.bin");
        Path stringModelPath = tempDir.resolve("string.bin");
        Path intModelPath = tempDir.resolve("int.bin");
        Path floatModelPath = tempDir.resolve("float.bin");

        // Create language model
        NeuralNet languageNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.inputSequenceEmbedding(10, 100, 16))
            .layer(Layers.hiddenGruLast(32))
            .output(Layers.outputSoftmaxCrossEntropy(100));
        SimpleNet.ofLanguageModel(languageNet).save(languageModelPath);
        
        // Create string classifier
        NeuralNet stringNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        SimpleNet.ofStringClassification(stringNet).save(stringModelPath);
        
        // Create int classifier
        NeuralNet intNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        SimpleNet.ofIntClassification(intNet).save(intModelPath);
        
        // Create float regressor
        NeuralNet floatNet = NeuralNet.newBuilder()
            .input(100)
            .setDefaultOptimizer(new SgdOptimizer(0.01f))
            .layer(Layers.hiddenDenseRelu(50))
            .output(Layers.outputLinearRegression(1));
        SimpleNet.ofFloatRegression(floatNet).save(floatModelPath);

        // Test all invalid cross-type loading combinations
        
        // Language model cannot be loaded as other types
        assertThrows(IOException.class, () -> SimpleNetString.load(languageModelPath));
        assertThrows(IOException.class, () -> SimpleNetInt.load(languageModelPath));
        assertThrows(IOException.class, () -> SimpleNetFloat.load(languageModelPath));

        // String model cannot be loaded as other types
        assertThrows(IOException.class, () -> SimpleNetLanguageModel.load(stringModelPath));
        assertThrows(IOException.class, () -> SimpleNetInt.load(stringModelPath));
        assertThrows(IOException.class, () -> SimpleNetFloat.load(stringModelPath));

        // Int model cannot be loaded as other types
        assertThrows(IOException.class, () -> SimpleNetLanguageModel.load(intModelPath));
        assertThrows(IOException.class, () -> SimpleNetString.load(intModelPath));
        assertThrows(IOException.class, () -> SimpleNetFloat.load(intModelPath));

        // Float model cannot be loaded as other types
        assertThrows(IOException.class, () -> SimpleNetLanguageModel.load(floatModelPath));
        assertThrows(IOException.class, () -> SimpleNetString.load(floatModelPath));
        assertThrows(IOException.class, () -> SimpleNetInt.load(floatModelPath));
    }
}