package dev.neuronic.net.serialization;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.*;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.HuberRegressionOutput;
import dev.neuronic.net.outputs.LinearRegressionOutput;
import dev.neuronic.net.outputs.SigmoidBinaryCrossEntropyOutput;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Centralized serialization service that eliminates tight coupling between components.
 * 
 * <p><b>Problem it solves:</b> Individual classes (like output layers) should not need to know 
 * about every possible optimizer, layer, or activator type. This creates tight coupling and 
 * violates the open/closed principle.
 * 
 * <p><b>Solution:</b> Centralized registry with automatic discovery and delegation.
 * 
 * <p><b>Benefits:</b>
 * <ul>
 *   <li><b>Loose coupling:</b> Components only know about their immediate dependencies</li>
 *   <li><b>Extensibility:</b> New types can be added without modifying existing code</li>
 *   <li><b>Single responsibility:</b> Serialization logic centralized in one place</li>
 *   <li><b>Type safety:</b> Compile-time guarantees for registered types</li>
 * </ul>
 * 
 * <p><b>Usage:</b>
 * <pre>{@code
 * // Instead of each class having its own deserializeOptimizer method:
 * Optimizer optimizer = SerializationService.deserializeOptimizer(in, typeId, version);
 * 
 * // Automatic registration at startup:
 * SerializationService.registerBuiltinTypes();
 * }</pre>
 */
public final class SerializationService {
    
    /**
     * Factory interface for creating serializable components from streams.
     */
    @FunctionalInterface
    public interface DeserializationFactory<T> {
        T deserialize(DataInputStream in, int version) throws IOException;
    }
    
    /**
     * Factory interface for creating layers that need FastRandom during deserialization.
     */
    @FunctionalInterface
    public interface LayerDeserializationFactory {
        Layer deserialize(DataInputStream in, int version, FastRandom random) throws IOException;
    }
    
    // Type registries - using concurrent maps for thread safety
    private static final Map<Integer, DeserializationFactory<Optimizer>> optimizerFactories = new ConcurrentHashMap<>();
    private static final Map<Integer, LayerDeserializationFactory> layerFactories = new ConcurrentHashMap<>();
    private static final Map<Class<?>, Integer> typeIdMapping = new ConcurrentHashMap<>();
    
    // Static initialization flag
    private static volatile boolean initialized = false;
    
    private SerializationService() {} // Prevent instantiation
    
    /**
     * Initialize the service with built-in type mappings.
     * Thread-safe and idempotent.
     */
    public static void initialize() {
        if (initialized) return;
        
        synchronized (SerializationService.class) {
            if (initialized) return;
            
            registerBuiltinOptimizers();
            registerBuiltinLayers();
            
            initialized = true;
        }
    }
    
    /**
     * Register all built-in optimizer types.
     * Eliminates the need for each component to know about every optimizer.
     */
    private static void registerBuiltinOptimizers() {
        // Register SGD optimizer
        registerOptimizer(SerializationConstants.TYPE_SGD_OPTIMIZER, 
            SgdOptimizer.class,
            SgdOptimizer::deserialize);
            
        // Register Adam optimizer  
        registerOptimizer(SerializationConstants.TYPE_ADAM_OPTIMIZER,
            AdamOptimizer.class,
            AdamOptimizer::deserialize);
            
        // Register AdamW optimizer
        registerOptimizer(SerializationConstants.TYPE_ADAMW_OPTIMIZER,
            AdamWOptimizer.class,
            AdamWOptimizer::deserialize);
    }
    
    /**
     * Register all built-in layer types.
     */
    private static void registerBuiltinLayers() {
        // Register output layers
        registerLayer(SerializationConstants.TYPE_SOFTMAX_CROSSENTROPY_OUTPUT,
            SoftmaxCrossEntropyOutput.class,
            (in, version, random) -> SoftmaxCrossEntropyOutput.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_LINEAR_REGRESSION_OUTPUT,
            LinearRegressionOutput.class,
            (in, version, random) -> LinearRegressionOutput.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_HUBER_REGRESSION_OUTPUT,
            HuberRegressionOutput.class,
            (in, version, random) -> HuberRegressionOutput.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_SIGMOID_BINARY_OUTPUT,
            SigmoidBinaryCrossEntropyOutput.class,
            (in, version, random) -> SigmoidBinaryCrossEntropyOutput.deserialize(in, version, random));
            
        // Register input/hidden layers
        registerLayer(SerializationConstants.TYPE_DENSE_LAYER,
            DenseLayer.class,
            (in, version, random) -> DenseLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_INPUT_EMBEDDING_LAYER,
            InputEmbeddingLayer.class,
            (in, version, random) -> InputEmbeddingLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_MIXED_FEATURE_INPUT_LAYER,
            MixedFeatureInputLayer.class,
            (in, version, random) -> MixedFeatureInputLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_GRU_LAYER,
            GruLayer.class,
            (in, version, random) -> GruLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_INPUT_SEQUENCE_EMBEDDING_LAYER,
            InputSequenceEmbeddingLayer.class,
            (in, version, random) -> InputSequenceEmbeddingLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_DROPOUT_LAYER,
            DropoutLayer.class,
            (in, version, random) -> DropoutLayer.deserialize(in, version, random));
            
        registerLayer(SerializationConstants.TYPE_LAYER_NORM_LAYER,
            LayerNormLayer.class,
            (in, version, random) -> LayerNormLayer.deserialize(in, version, random));
    }
    
    /**
     * Register an optimizer type with the service.
     */
    public static void registerOptimizer(int typeId, Class<? extends Optimizer> clazz, 
                                       DeserializationFactory<Optimizer> factory) {
        optimizerFactories.put(typeId, factory);
        typeIdMapping.put(clazz, typeId);
    }
    
    /**
     * Register a layer type with the service.
     */
    public static void registerLayer(int typeId, Class<? extends Layer> clazz,
                                   LayerDeserializationFactory factory) {
        layerFactories.put(typeId, factory);
        typeIdMapping.put(clazz, typeId);
    }
    
    /**
     * Deserialize an optimizer using the centralized registry.
     * Eliminates the need for each component to have its own deserializeOptimizer method.
     * 
     * @param in input stream
     * @param typeId optimizer type ID
     * @param version serialization version
     * @return deserialized optimizer
     * @throws IOException if deserialization fails
     */
    public static Optimizer deserializeOptimizer(DataInputStream in, int typeId, int version) throws IOException {
        ensureInitialized();
        
        DeserializationFactory<Optimizer> factory = optimizerFactories.get(typeId);
        if (factory == null) {
            throw new IOException("Unknown optimizer type ID: " + typeId + 
                                ". Available types: " + optimizerFactories.keySet());
        }
        
        return factory.deserialize(in, version);
    }
    
    /**
     * Deserialize a layer using the centralized registry.
     * 
     * @param in input stream
     * @param typeId layer type ID  
     * @param version serialization version
     * @param random FastRandom instance from parent network
     * @return deserialized layer
     * @throws IOException if deserialization fails
     */
    public static Layer deserializeLayer(DataInputStream in, int typeId, int version, FastRandom random) throws IOException {
        ensureInitialized();
        
        LayerDeserializationFactory factory = layerFactories.get(typeId);
        if (factory == null) {
            throw new IOException("Unknown layer type ID: " + typeId + 
                                ". Available types: " + layerFactories.keySet());
        }
        
        return factory.deserialize(in, version, random);
    }
    
    /**
     * Get the type ID for a serializable object.
     * 
     * @param obj the object to get type ID for
     * @return type ID, or null if not registered
     */
    public static Integer getTypeId(Object obj) {
        ensureInitialized();
        return typeIdMapping.get(obj.getClass());
    }
    
    /**
     * Write a serializable object with its type ID.
     * 
     * @param out output stream
     * @param obj the object to serialize
     * @param version serialization version
     * @throws IOException if serialization fails
     */
    public static void writeWithTypeId(DataOutputStream out, Serializable obj, int version) throws IOException {
        Integer typeId = getTypeId(obj);
        if (typeId == null) {
            throw new IOException("Object type not registered for serialization: " + obj.getClass());
        }
        
        out.writeInt(typeId);
        obj.writeTo(out, version);
    }
    
    /**
     * Check if the service supports a given type ID.
     */
    public static boolean supportsOptimizer(int typeId) {
        ensureInitialized();
        return optimizerFactories.containsKey(typeId);
    }
    
    /**
     * Check if the service supports a given type ID.
     */
    public static boolean supportsLayer(int typeId) {
        ensureInitialized();
        return layerFactories.containsKey(typeId);
    }
    
    /**
     * Get information about registered types (for debugging).
     */
    public static String getRegistrationInfo() {
        ensureInitialized();
        
        StringBuilder sb = new StringBuilder();
        sb.append("SerializationService Registration Info:\n");
        sb.append("Optimizers: ").append(optimizerFactories.keySet()).append("\n");
        sb.append("Layers: ").append(layerFactories.keySet()).append("\n");
        sb.append("Type mappings: ").append(typeIdMapping.size()).append(" classes registered\n");
        
        return sb.toString();
    }
    
    /**
     * Ensure the service is initialized before use.
     */
    private static void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }
}