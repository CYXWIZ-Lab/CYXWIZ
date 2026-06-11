# tofix23 - Third-Party Node Plugins and Optional Text Extensions

## Purpose

Document future work that is intentionally outside the current native `tofix11`
text-node scope.

The main design target is to let CyxWiz Engine use third-party node
implementations through the plugin system, so optional or heavyweight node
features do not have to live directly inside the core engine.

## Background

`tofix11` implemented the native text path for:

- `TextTokenizer`
- `TextVocabulary`
- `TextPadding`
- `Embedding`

That gives the engine a working built-in flow for tokenization, vocabulary
build/load/save, padding, and embedding configuration.

Some text features are still better treated as optional extensions because they
can carry extra dependencies, model files, licensing questions, or implementation
complexity. Examples include SentencePiece, Hugging Face tokenizers, external
embedding providers, and custom preprocessing libraries.

## Future Scope

### Optional Text/Vocabulary Extensions

Candidate extension nodes:

- `TextVocabulary_3prt`
- `Tokenizer_3prt`
- `EmbeddingProvider_3prt`
- `TextNormalizer_3prt`

Possible third-party implementations:

- SentencePiece
- Hugging Face tokenizers
- spaCy tokenization
- WordPiece/BPE libraries
- FastText/GloVe loaders
- domain-specific tokenizers for code, genomics, finance, or medical text

These should not block the native text nodes. The native nodes should remain the
stable fallback path that works without external plugins.

## Third-Party Node Plugin Design

### Goal

Allow external plugins to provide real node implementations that appear in the
CyxWiz node editor, compile into executable graph steps, and run during training
or preprocessing.

The plugin should be able to define:

- Node metadata
- Input/output pins
- Property schema
- Optional custom configuration UI
- Runtime execution behavior
- Optional compiler integration
- Serialization contract

### Node Registration

Plugins should register node types at load time through the existing plugin
manager.

Required metadata:

- Stable plugin id
- Stable node type id
- Display name
- Category
- Version
- Description
- Input pin schema
- Output pin schema
- Property schema
- Runtime capability flags

Example capability flags:

- `preprocess`
- `layer`
- `loss`
- `optimizer`
- `metric`
- `visualization`
- `debug_only`
- `gpu_supported`
- `cpu_only`

### Property Schema

Plugin nodes need a structured property schema so the engine can render a safe
default editor even when the plugin does not provide custom UI.

Supported property types should include:

- string
- int
- float
- bool
- enum
- file path
- directory path
- column selector
- tensor shape
- JSON object

Each property should define:

- key
- label
- type
- default value
- required flag
- validation rule
- tooltip/help text

### Custom Dialogs

Plugins may optionally provide a richer config dialog.

If no custom dialog exists, CyxWiz should render a generic dialog from the
property schema.

This mirrors the native direction used by nodes like `Embedding`, where some
nodes need more configuration space than the compact Properties panel can
reasonably provide.

### Runtime Execution

Plugin node runtime should support at least two execution modes:

- Native C++ plugin execution
- External process execution

Native execution is best for high-performance or tightly integrated nodes.
External process execution is safer for Python-heavy or dependency-heavy nodes.

The first practical implementation should prioritize one safe path instead of
trying to solve every execution mode at once.

Recommended first target:

- Preprocessing operator plugins
- Table/text/tensor input and output contracts
- CPU execution
- Clear error reporting

GPU support should come later after lifecycle, memory ownership, and device
selection rules are stable.

### Compiler Integration

Graph compilation needs a plugin node boundary.

The compiler should be able to ask the plugin registry:

- Is this node type known?
- What runtime contract does it expose?
- Can it compile for the current graph mode?
- What validation errors should block execution?
- What warnings should be shown to the user?

For model-layer plugins, the compiler also needs:

- Tensor shape inference
- Parameter count reporting
- Forward/backward capability
- Serialization of trainable state
- Device compatibility

This is larger than preprocessing plugins and should be a later phase.

### Serialization

Saved graphs must preserve plugin node identity and configuration.

Graph JSON should store:

- plugin id
- plugin version
- node type id
- node display name
- property values
- schema version

On load, if a plugin is missing, CyxWiz should:

- Keep the node visible in the graph
- Mark it as unresolved
- Preserve its saved config
- Block execution with a clear error
- Offer a plugin install/load hint when possible

### Error Reporting

Plugin node failures need vivid user-facing errors.

Errors should explain:

- Which plugin failed
- Which node failed
- What phase failed: load, validate, compile, execute, serialize
- What the user can do next

This is important because third-party nodes will fail in ways core nodes do not:
missing DLLs, incompatible plugin versions, missing Python packages, model file
not found, license-restricted assets, or unsupported GPU/runtime capabilities.

## Implementation Phases

### Phase 1 - Design Contract

- Define plugin node metadata schema
- Define property schema
- Define serialization format
- Define unresolved plugin behavior
- Define compiler validation hooks

### Phase 2 - Preprocessing Plugin Nodes

- Register plugin nodes in the node browser
- Render generic property dialogs from schema
- Execute CPU preprocessing plugins
- Add logs and Studio Debugger traces
- Add missing-plugin preservation in saved graphs

### Phase 3 - Text Extension Plugins

- Implement `Tokenizer_3prt`
- Implement `TextVocabulary_3prt`
- Support SentencePiece or another focused tokenizer first
- Support external vocab/model files
- Keep native tokenizer/vocabulary as fallback

### Phase 4 - Model/Layer Plugins

- Add plugin-provided model layers
- Add shape inference hooks
- Add trainable parameter serialization
- Add CPU/GPU capability checks
- Add forward/backward runtime contracts

## Open Questions

- Should plugin nodes be allowed to run inside the engine process, or should
  dependency-heavy plugins run out-of-process by default?
- How strict should plugin version compatibility be when opening old graphs?
- Should plugin node schemas be C++ structs, JSON manifests, or both?
- How should plugin-provided custom dialogs be sandboxed?
- What is the minimum runtime contract for trainable layer plugins?
- Should Python plugin nodes use the existing Python environment settings or
  require per-plugin environments?

## Recommendation

Do not implement this inside `tofix11`.

Use `tofix23` as the design bucket for third-party node implementation support.
The first real implementation should be narrow:

- Plugin-provided preprocessing node
- Generic schema-driven config dialog
- Saved graph preservation
- Clear compile/runtime errors

After that works, add SentencePiece or another tokenizer as the first concrete
third-party text node.
