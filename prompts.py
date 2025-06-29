class prompts: 


    FIRST_GENERATION_TEMPLATE = f"""
    You are a highly creative and rigorous algorithmic thinker. Your reasoning and theme is guided by {THEME}, and your goal is to create a novel, low-level implementation of the following task:

    Task:
    {TASK_DESCRIPTION}

    Guidance:
    - Do not rely on conventional textbook methods unless extremely radically reinterpreted.
    - Explore unorthodox ideas. It's okay if they seem impractical or speculative.
    - Optimize for minimal steps, instruction count, or CPU cycles where appropriate.
    - Only generate code in {LANGUAGE}, Don't use inline comments.
    - Prioritize **novel algorithmic structure**, not just minor speedups.
    - Think as if you are reinventing computation from a new paradigm.
    - You are encouraged to borrow metaphors from the theme: try abstracting the task as a problem in your domain experience.
    - The more unexpected your method, the better. Avoid known optimizations.
    - Imagine you're inventing this for a future computing system.
    """

    CHILD_TEMPLATE = f"""
        # EVOLVE-BLOCK-START
        {PARENT_CODE}
        # EVOLVE-BLOCK-END

        Prior evaluation:
        {PARENT_EVAL_SUMMARY}

        Task:
        Given the above implementation, **propose a divergent variant** that further
        reduces instruction count or CPU cycles for this routine, while preserving
        correctness. You may also restructure loops, unroll, or exploit register-level
        parallelism—even if it seems unconventional.

        Guidelines:
        - Your reasoning and theme is guided by {THEME}
        - **Use only x86-64 assembly**, NASM or AT&T syntax (be consistent).
        - Return your answer as a sequence of diff blocks using the exact format below.
        - Do **not** rewrite the entire function—only specify the minimal changes.
        - Avoid known textbook tricks unless you reinterpret them in a new way.
        - Novelty is more important than marginal gains.

        """
    
    MARRY_TEMPLATE = f"" ""


    themes = [
        # Quantum Computing & Physics (33 themes)
        "Google Willow's surface code distance scaling achieves exponential error suppression with Λ factors of 2.14+ as physical qubit count increases. Apply hierarchical error-correcting data structures where adding redundant elements exponentially improves reliability in memory systems.",
        "Topological Majorana zero modes provide hardware-level error protection through exotic quasiparticles that are their own antiparticles. Design self-correcting algorithms where computational elements inherently resist corruption through topological data representations.",
        "Bivariate bicycle LDPC codes achieve distance-12 quantum error correction with only 144 physical qubits per logical qubit. Create sparse matrix algorithms with built-in error correction using minimal overhead for high-performance computing.",
        "Real-time quantum error decoding processes error syndromes in sub-microsecond 1.1μs cycle times for superconducting processors. Implement real-time adaptive algorithms that correct computational errors on-the-fly during execution without stopping main threads.",
        "Cat qubit stabilization reduces error correction from 2D to 1D problems achieving 99.9999% fidelity with fewer resources. Apply dimension-reduction techniques for complex optimization problems by encoding solutions in lower-dimensional spaces.",
        "Counterdiabatic driving optimization uses adiabatic gauge potentials to suppress unwanted quantum transitions while accelerating evolution. Design momentum-based optimization algorithms that use gauge fields to prevent premature convergence in gradient descent.",
        "QAOA scaling demonstrates 54-qubit, 4-layer implementations with 324 RZZ gates achieving 94%+ fidelity. Create layered approximation algorithms where solution quality improves systematically with computational depth.",
        "Quantum amplitude interference patterns use constructive and destructive interference to amplify correct solutions while canceling incorrect ones. Design algorithms that use wave-like interference of computational paths to naturally converge on optimal solutions.",
        "Superposition-based quantum parallelism processes 2^n states simultaneously in n-qubit systems through quantum superposition. Implement massively parallel algorithm architectures that exploit implicit parallelism in data structures and computational paths.",
        "Quantum Maxwell's demon converts quantum information into work through stimulated emission in microwave cavities. Create information-to-energy conversion algorithms that extract computational work from data correlations and statistical patterns.",
        "Double quantum dot information engines use quantum coherence to guide electrons against voltage bias through energy level tuning. Design energy-aware algorithms that use system state information to perform work more efficiently than thermodynamic limits suggest.",
        "Time-periodic quantum phase transitions use abrupt periodic parameter jumps instead of continuous variations to study phase transitions. Create algorithms that use periodic parameter switching to explore solution spaces more efficiently than continuous optimization.",
        "Measurement-induced phase transitions occur at critical measurement rates that cause transitions in entanglement structure. Design adaptive sampling algorithms that transition between different computational regimes based on observation density.",
        "Magnetohydrodynamic plasma simulation uses hybrid PIC-MHD models balancing particle-level detail with macroscopic fluid behavior. Create multi-scale algorithms that seamlessly transition between fine-grained and coarse-grained computational models.",
        "Quantum enhanced plasma instability control uses QAOA to control stability criteria in tokamak plasmas. Design stability-aware algorithms that use optimization techniques to maintain system stability in dynamic environments.",
        "Noise-enhanced quantum teleportation uses multipartite hybrid entanglement where noise actually improves teleportation fidelity. Create noise-assisted algorithms that use controlled randomness to improve information transfer efficiency.",
        "Coexistent classical-quantum communication achieves 30km quantum teleportation through busy fiber optic cables carrying internet traffic. Design hybrid communication protocols that multiplex different information streams without interference.",
        "Distributed quantum processor networks link independent quantum processors through photonic interfaces to create unified systems. Create distributed computing architectures that seamlessly link independent processing units through optical interconnects.",
        "Entanglement swapping protocols create entanglement between particles that never directly interacted. Design transitive connection algorithms that establish relationships between computational elements through intermediate connections.",
        "Variational quantum eigensolver acceleration uses hybrid quantum-classical optimization for finding ground states with improved convergence. Create hybrid optimization algorithms that combine quantum-inspired parallel search with classical refinement steps.",
        "Quantum phase estimation for spectral analysis enables fast eigenvalue determination for unitary operators. Design spectral algorithms that rapidly extract eigenvalue information from large matrices through phase estimation techniques.",
        "Logic gate universality through error correction uses magic state distillation enabling universal quantum computation with error-corrected logical qubits. Create universal algorithm frameworks that achieve full computational expressivity through error-corrected primitive operations.",
        "Bosonic energy subtraction schemes use Jaynes-Cummings systems for subtracting bosonic energy inferred from qubit measurements. Design energy-efficient algorithms that selectively remove computational overhead by measuring and responding to system states.",
        "Landauer limit-aware computing recognizes that information processing requires minimum energy of kbT ln(2) per bit operation. Create ultra-low-power algorithms that approach fundamental thermodynamic limits through reversible computation techniques.",
        "Hybrid information-thermodynamic cycles extract work from quantum systems through information feedback in cyclical processes. Design cyclical optimization algorithms that use information gained in previous iterations to perform more efficient computation in subsequent cycles.",
        "Many-body localization transitions prevent systems from thermalizing due to strong disorder, preventing ergodic behavior. Create disorder-resistant algorithms that maintain computational coherence even in highly noisy environments.",
        "Electromagnetic field quantum simulations model wave-particle interactions in complex electromagnetic fields. Design field-aware algorithms that account for wave-particle duality in distributed computing systems.",
        "Plasma wave dispersion control uses quantum lattice-based methods enabling greater experimental control of plasma processes. Create wave-based algorithms that use dispersion relations to control information propagation in distributed systems.",
        "Nonlinear plasma behavior modeling handles inherent nonlinearity from electromagnetic particle interactions causing turbulence and instabilities. Design nonlinear-aware algorithms that expect and exploit chaotic behavior rather than trying to suppress it.",
        "Quantum memory multiplexing uses multiplexed quantum memories to enable long-distance teleportation protocols. Create memory-multiplexed algorithms that use multiple storage systems to enable long-range data dependencies.",
        "Diamond quantum bit strain engineering stretches diamond films to create quantum bits with reduced equipment requirements. Design adaptive data structure algorithms that dynamically adjust memory layouts to optimize performance with minimal hardware overhead.",
        "Phonon quantum beamsplitting splits sound waves to demonstrate quantum properties and create superposition states. Create acoustic-inspired algorithms that use wave-splitting techniques for parallel signal processing and data routing.",
        "Dynamical quantum phase transitions via Loschmidt echo show third-order phase transitions in quantum spin chains with exponential finite-size scaling. Design echo-based algorithms that detect computational phase transitions through fidelity measurements.",

        # Neuroscience & AI (33 themes)
        "SparseK attention with differentiable top-K selection uses a scoring network and differentiable top-k mask operator to select constant key-value pairs for each query. Create adaptive filtering algorithms that dynamically select the most relevant data points for processing with linear time complexity.",
        "Semantic focus sparse attention concentrates computational resources on semantically relevant tokens with input-dependent sparsity patterns. Design priority-based processing systems that allocate computational resources based on semantic importance and faster convergence.",
        "Dozer attention for time series combines local, stride, and vary components where vary dynamically expands historical context based on forecasting horizons. Create adaptive memory systems that extend their temporal context based on prediction requirements.",
        "Spark transformer with dual sparsity achieves high activation sparsity in both feed-forward and attention mechanisms while maintaining model quality. Design energy-efficient algorithms that minimize active computational units without performance degradation.",
        "Temporal predictive coding networks implement biologically plausible predictive coding using local Hebbian plasticity and recurrent connections. Create online learning algorithms that continuously adapt predictions using only local information.",
        "Hierarchical predictive coding with generalized coordinates extends predictive coding to multiple temporal derivatives enabling multi-scale prediction. Design multi-scale prediction algorithms for complex dynamical systems at different time scales.",
        "Dynamic predictive coding for sequence learning generates diverse activity patterns at higher levels while selectively reinforcing high-value patterns. Create hierarchical reinforcement learning algorithms with dynamic pattern generation capabilities.",
        "Simulation-selection model for memory consolidation uses CA3 to generate diverse activity patterns while CA1 selectively reinforces high-value patterns. Design selective data retention algorithms that prioritize valuable information patterns for long-term storage.",
        "Attention-based dynamic neuromorphic computing implements attention mechanisms in spiking networks with dynamic imbalance detection. Create power-efficient attention algorithms that activate only when needed through event-driven processing.",
        "Neuromorphic backpropagation on Loihi implements exact backpropagation algorithm entirely on neuromorphic chips using synfire-gated dynamical coordination. Design fully distributed learning algorithms that operate without centralized coordination mechanisms.",
        "Spatio-temporal pruning for SNNs adaptively prunes both spatial connections and temporal dimensions based on synaptic operations and decision timesteps. Create dynamic algorithm optimization that reduces computational complexity across multiple dimensions simultaneously.",
        "Constructive community race SNNs use full-density spiking networks with competitive dynamics between neural communities for efficient computation. Design competitive computational frameworks where multiple processing units compete for limited resources.",
        "Generative replay for category learning uses generative models to replay category instances during consolidation, selectively strengthening weaker memories. Create adaptive training algorithms that focus computational resources on difficult or poorly learned patterns.",
        "Reinforcement learning-guided replay implements an RL system that learns which memories to replay based on performance contribution. Design meta-learning algorithms that optimize their own training strategies through experience-based selection.",
        "Corticohippocampal hybrid networks combine artificial and spiking neural networks to emulate dual memory representations preventing catastrophic forgetting. Create hybrid learning systems that maintain both specific and generalized knowledge representations.",
        "Awake replay selection mechanisms use hippocampal sharp wave ripples to selectively replay high-value experiences during awake states. Create selective experience replay algorithms that prioritize valuable training examples for memory consolidation.",
        "Biologically detailed V1 modeling uses large-scale V1 models with comprehensive anatomical data demonstrating robust visual processing across multiple tasks. Design multi-layered visual processing algorithms with specialized functional modules for different visual tasks.",
        "Predictive visual cortical pathways model evolution from simple dual pathways to complex heterarchical networks with recurrent connections. Create adaptive visual processing pipelines that reorganize based on task requirements and environmental constraints.",
        "Apparent motion reconstruction in V1 reconstructs intermediate visual features during apparent motion that were never presented to the retina. Create interpolation algorithms that generate intermediate states for smooth transitions between discrete data points.",
        "Noise-robust neural coding restricts noise fluctuations to dimensions orthogonal to stimulus encoding enhancing coding robustness. Design robust encoding schemes that separate signal from noise in orthogonal dimensions for improved reliability.",
        "Adaptive moving self-organizing maps allow dynamic adjustment of neuron positions and addition/deletion of neurons during training. Create adaptive clustering algorithms that modify their structure based on changing data characteristics and requirements.",
        "Multi-view clustering with adaptive SOMs processes high-dimensional sparse data with diverse representations using subspace clustering approaches. Design multi-perspective data analysis algorithms that handle heterogeneous data sources with different representations.",
        "Topological extension SOMs explore alternative topologies derived from geometric tessellation theory for improved clustering performance. Create flexible clustering algorithms that adapt their topology to data geometry and optimize cluster boundaries.",
        "Sparsely coded boundary cycle SOMs use sparse boundary cycles to represent input patterns creating robust and unique representations. Design sparse coding algorithms that use geometric boundaries for efficient and distinctive pattern representation.",
        "Fisher information-based synaptic pruning estimates synaptic importance using locally available information combining weight magnitude with firing rates. Create information-theoretic pruning algorithms that optimize network topology based on local statistical measures.",
        "Developmental pruning with correlation-sensitive signals uses calcium-based correlation detection for homeostatic pruning effects combined with structural plasticity. Design self-regulating algorithms that maintain optimal network structure through local correlation detection mechanisms.",
        "Adaptive temporal pruning regulates pruning speed by modifying threshold functions of state transitions for effective sparsification. Create dynamic optimization algorithms that adjust their pruning strategies based on current system state and performance.",
        "Microglia-inspired synaptic pruning models microglial engulfment of redundant synapses for network optimization during development. Design distributed maintenance algorithms that remove redundant connections through local cleanup processes.",
        "Sensorimotor contingency learning acquires new sensorimotor mappings through sensory augmentation forming new contingency relationships. Create adaptive interface algorithms that learn new input-output mappings through experiential learning.",
        "Embodied executive function development integrates movement and cognitive processing where motor development drives cognitive capabilities. Design algorithms that use physical simulation to enhance abstract reasoning and problem-solving capabilities.",
        "SOVEREIGN2 visuomotor architecture integrates sensory, motor, and memory components with resonant dynamics for emergent cognitive properties. Create integrated perception-action systems with emergent behavioral capabilities through component interaction.",
        "Neuromorphic optical processing combines optical spectrum slicing with reconfigurable processors for efficient neural computation. Design optical computing algorithms that leverage wavelength-based parallel processing for enhanced computational efficiency.",
        "Three-factor learning in SNNs combines pre-synaptic, post-synaptic, and modulatory signals for enhanced learning in spiking networks. Create multi-signal learning algorithms that integrate multiple information sources for improved adaptation and performance.",

        # Computer Architecture (33 themes)
        "Attacker-controlled prediction mitigation uses new branch prediction schemes that detect and mitigate adversarial manipulation of branch target buffers. Design secure algorithm execution where computational paths are authenticated against tampering using cryptographic signatures.",
        "Confidence-gated pipeline execution dynamically stalls execution when branch prediction confidence falls below threshold reducing speculative power consumption. Create adaptive computation algorithms where uncertainty triggers conservative execution modes to maintain reliability.",
        "Multi-level speculative runahead enhances runahead execution that pre-processes instructions during cache misses using nested speculation levels. Design algorithmic patterns for multi-stage lookahead in data processing pipelines with hierarchical speculation.",
        "Transient execution recovery protocols provide hardware mechanisms for cleanly recovering from speculative execution attacks like Spectre/Meltdown. Create algorithm designs with rollback capabilities and state isolation mechanisms for security resilience.",
        "Indirect branch target prediction with machine learning uses AI-driven prediction of indirect branch targets using program context analysis. Create adaptive algorithm optimization based on runtime behavior patterns and learned execution preferences.",
        "Vector extension dynamic scheduling adapts RISC-V vector instruction scheduling to available functional units in real-time. Design algorithmic approaches for dynamic resource allocation in parallel processing with adaptive load balancing.",
        "Instruction set architecture profiling uses RVA23 profile-based adaptive instruction selection optimizing for specific workload characteristics. Create self-optimizing algorithms that adapt their instruction sequences based on computational workload analysis.",
        "Five-stage pipeline hazard prediction performs advanced hazard detection that predicts data/control dependencies multiple cycles ahead. Create algorithmic dependency analysis for parallel execution optimization with predictive conflict resolution.",
        "Speculative forwarding networks enhance data bypassing that speculatively forwards results through multiple pipeline stages. Design algorithmic patterns for predictive data flow in processing pipelines with speculative result propagation.",
        "Custom extension integration enables dynamic integration of custom RISC-V extensions based on workload analysis. Create adaptive algorithm design where specialized operations are dynamically enabled based on computational requirements.",
        "Hierarchical partial reconfiguration uses nested reconfigurable regions where child partitions exist within parent partitions. Design hierarchical algorithm structures with nested adaptive components that can be independently reconfigured.",
        "Dynamic classifier selection on FPGAs creates hardware that dynamically selects between different ML classifiers based on incoming data characteristics. Create adaptive algorithm switching based on input patterns and data characteristics.",
        "Real-time bitstream scheduling intelligently schedules partial reconfiguration bitstreams to minimize latency while maintaining functionality. Design just-in-time algorithm component loading with optimized resource scheduling and minimal overhead.",
        "Fault-tolerant reconfiguration uses partial reconfiguration that isolates and replaces faulty modules while maintaining system operation. Create self-healing algorithmic approaches with component isolation and automatic fault recovery mechanisms.",
        "Multi-objective evolutionary reconfiguration uses genetic algorithms to optimize FPGA resource allocation during runtime reconfiguration. Create evolutionary approaches to algorithm optimization with multi-objective fitness functions and adaptive resource allocation.",
        "Reinforcement learning enhanced fixing uses RL-based iterative program repair that improves code generation through feedback loops. Create self-improving algorithmic approaches with error correction and automated optimization through experience.",
        "Process reward model integration provides line-by-line code evaluation using learned reward models for program synthesis. Create fine-grained algorithmic quality assessment techniques with learned evaluation metrics and continuous improvement.",
        "Enumerative LLM-guided synthesis combines traditional enumerative synthesis with LLM probabilistic guidance for program generation. Create hybrid approaches combining exhaustive search with learned heuristics for efficient solution exploration.",
        "Syntax-guided synthesis with context uses program synthesis with grammar constraints and semantic context awareness. Create structured algorithm generation with domain-specific constraints and contextual optimization.",
        "Multi-agent code generation uses collaborative synthesis with multiple AI agents having different specializations. Create distributed algorithmic problem-solving approaches with specialized agents and collaborative optimization.",
        "Dynamic partial reconfiguration for side-channel defense uses real-time FPGA reconfiguration to disrupt power analysis attacks. Create algorithmic approaches that dynamically alter their execution patterns to prevent analysis and maintain security.",
        "Electromagnetic fault injection countermeasures provide hardware mechanisms that detect and counter EM-based fault injection attacks. Create algorithmic fault detection and recovery mechanisms with electromagnetic interference resistance.",
        "Pluton-style updateable security processors use Rust-based security processors with memory-safe firmware that can be updated dynamically. Create secure algorithm design with updateable components and memory-safe execution environments.",
        "Machine learning-based glitch detection uses AI systems that detect timing-based attacks by analyzing clock signal patterns. Create algorithmic approaches for anomaly detection in execution patterns with ML-based security monitoring.",
        "Cryptographic instruction set extensions provide hardware-accelerated cryptographic operations with built-in side-channel resistance. Create algorithmic approaches that integrate security operations natively with hardware-accelerated cryptographic primitives.",
        "Heterogeneous cache coherence protocols use unified coherence protocols for mixed CPU/GPU/accelerator systems with common internal protocols. Create algorithmic approaches for heterogeneous system coordination with unified communication protocols.",
        "Directory-based coherence with shared cache locates coherence directories in shared L3 cache rather than main memory. Create algorithmic approaches for distributed coordination with local caching and improved memory access efficiency.",
        "Dual-consistency cache protocols simultaneously provide strong and weak consistency models based on program region requirements. Create adaptive consistency algorithms that provide different consistency guarantees based on application requirements.",
        "Snoop filter optimization uses advanced snoop filters with presence vectors and temporal algorithms for cache line replacement. Create algorithmic approaches for intelligent resource replacement with temporal locality optimization.",
        "MESIF protocol extensions enhance cache coherence states that optimize for specific workload patterns. Create algorithmic state management with workload-specific optimizations and adaptive cache behavior.",
        "Ternary quantum dot cellular automata synthesis performs logic synthesis for three-valued quantum systems using modified Karnaugh maps. Create multi-valued algorithmic approaches beyond binary logic with ternary state optimization.",
        "Quantum circuit minimization with Monte Carlo graph search uses stochastic optimization of quantum gate sequences with probabilistic search. Create algorithmic approaches for complex optimization problems using Monte Carlo methods and graph-based search.",
        "Analog-digital hybrid processing uses circuits that seamlessly integrate analog computation with digital control for energy-efficient processing. Create algorithmic approaches that combine continuous and discrete computation for enhanced efficiency.",

        # Mathematics & Theory (33 themes)
        "Linear homotopy type theory extends HoTT with dependent linear types enabling quantum data type certification by combining classical HoTT with resource-aware computation. Create algorithms for quantum circuit verification with topological protection and resource tracking.",
        "Internal parametricity without intervals uses BCH cubes instead of interval types for univalence in type theory implementations. Create new algorithmic paradigms for equality checking and program transformation that avoid complex interval arithmetic.",
        "Higher observational type theory defines identity types by recursion on base types enabling more efficient algorithmic identity checking. Create efficient identity verification algorithms by leveraging structural recursion patterns in low-level implementations.",
        "Synthetic cubical homotopy theory enables computational proofs in synthetic homotopy theory through cubical Agda developments. Create algorithms that automatically verify topological properties through direct computation rather than symbolic manipulation.",
        "Topological quantum gates in HoTT show quantum logic gates as transport in Eilenberg-MacLane types for quantum computation. Create algorithms for quantum gate synthesis using homotopy-theoretic path optimization and topological constraints.",
        "Fibrant universe construction uses higher coinductive types to construct fibrant universes in type theory. Create algorithms for type checking with improved termination guarantees through coinductive structural analysis.",
        "Dynamic lifting in quantum monads enables classical control of quantum measurements through quantum monadology approaches. Create algorithms for hybrid quantum-classical computation with formal verification capabilities and monadic structure.",
        "Concurrent monads for shared state use parallel composition as a control structure for concurrent programming. Create algorithms for lock-free concurrent data structures using monadic composition laws and parallel execution.",
        "Procontainer theory extends container functors to profunctors enabling arrow-based computational effects. Create algorithms for streaming computation and incremental processing with categorical guarantees and compositional structure.",
        "Monoidal profunctor representations show Moore machines as lawful monoidal profunctors for stateful computation. Create algorithms for stateful stream processing with compositional correctness by construction and categorical semantics.",
        "Operadic DSL composition formalizes operads for categorical DSL frameworks and language composition. Create algorithms for type-safe language composition and embedded domain-specific compilation with operadic structure.",
        "Kleisli monoidal categories organize probability polymeasures for probabilistic computation with categorical structure. Create algorithms for probabilistic computation with compositional semantics and measure-theoretic soundness.",
        "Point-level topological features extract node-level topological features from point clouds for geometric analysis. Create algorithms for geometric feature detection using discrete differential geometry operators and topological invariants.",
        "Parametrized persistent homology advances multi-parameter TDA enabling hierarchical clustering with topological stability. Create algorithms for hierarchical clustering with topological stability guarantees across multiple scales simultaneously.",
        "Discrete Morse theory integration enables topological simplification with guaranteed homotopy preservation in mesh processing. Create algorithms for topological simplification with guaranteed homotopy preservation and computational efficiency.",
        "Alpha complex optimization improves alpha complex computation over Vietoris-Rips for persistent homology analysis. Create algorithms for more efficient persistent homology with reduced computational complexity and improved accuracy.",
        "Persistent spectral sequences connect persistent homology with spectral sequences for hierarchical data analysis. Create algorithms for hierarchical data analysis using algebraic topology filtrations and spectral sequence computations.",
        "Neural differential forms move beyond persistent homology to differential topology for graph learning applications. Create algorithms for graph learning using de Rham cohomology and differential form integration techniques.",
        "Cellular approximation in constructive HoTT advances cellular homology formalization with constructive guarantees. Create algorithms for computational homology with constructive guarantees and efficient normalization procedures.",
        "Riemannian flow matching uses flow matching with equivariances for generative modeling on manifolds. Create algorithms for generative modeling on manifolds with Riemannian geometric constraints and equivariant properties.",
        "Finsler geometry for stochastic manifolds compares expected lengths from stochastic manifolds for uncertainty quantification. Create algorithms for uncertainty quantification using Finsler metric structures and stochastic geometry.",
        "Geometric algebra networks extend GATr to multiple algebras for E(3)-equivariant neural networks. Create algorithms for E(3)-equivariant neural networks using Clifford algebra operations for 3D data processing.",
        "Differentiable molecular dynamics advances differentiable AIMD with ML functionals for materials simulation. Create algorithms for materials simulation with gradient-based optimization of interatomic potentials and differentiable physics.",
        "Information geometric VAE decoding applies Fisher-Rao metrics to VAE latent spaces for generative modeling. Create algorithms for generative modeling with information-theoretic distance preservation and geometric structure.",
        "Natural gradient manifold optimization uses differential geometry for non-convex optimization in neural network training. Create algorithms for neural network training with Riemannian gradient methods and manifold optimization.",
        "Quantum LDPC code implementation advances low-density parity-check codes for quantum error correction. Create algorithms for efficient quantum error decoding with improved thresholds and reduced computational overhead.",
        "Color code logical operations implement color codes for fault-tolerant quantum gate synthesis. Create algorithms for fault-tolerant quantum gate synthesis with more efficient logical operations than surface codes.",
        "AI-enhanced quantum decoders use machine learning decoders achieving 6% error reduction in quantum error correction. Create algorithms for adaptive quantum error correction using neural network state estimation and learning.",
        "Hardware-assisted bosonic codes develop cat qubits and dual-rail architectures for quantum error correction. Create algorithms for error correction with hardware-software co-design and reduced overhead through bosonic encoding.",
        "Cubical lambda calculus advances cubical type theory computational meaning for automated theorem proving. Create algorithms for automated theorem proving with built-in homotopy reasoning and efficient equality checking.",
        "Dependent linear types combine dependent types with linear logic for resource-aware functional programming. Create algorithms for resource-aware functional programming with compile-time memory management guarantees and linear type systems.",
        "Synthetic proof data generation creates 8 million formal proofs from competition problems for theorem proving. Create algorithms for large-scale automated theorem proving using synthetic training data and formal proof generation.",
        "Global batch fine-tuning for theorem proving achieves 46.3% accuracy on miniF2F for formal mathematics. Create algorithms for large language model training on formal mathematics with improved proof search strategies.",

        # Cryptography & Information Theory (33 themes)
        "ML-KEM module-lattice-based key encapsulation uses NIST's FIPS 203 standard with modular arithmetic operations and structured lattice reductions. Create efficient data organization and error-resilient indexing algorithms using lattice-based modular structures.",
        "ML-DSA module-lattice-based digital signatures use FIPS 204 standard with Fiat-Shamir abort methodology for quantum-resistant authentication. Create fault-tolerant algorithms that gracefully handle invalid states during computation using adaptive abort mechanisms.",
        "FN-DSA FALCON-derived signatures use fast Fourier transforms over NTRU lattices for efficient post-quantum signatures. Create frequency-domain approaches to data processing and pattern recognition using FFT-based lattice operations.",
        "Learning with errors problem variants refine LWE parameter selection for better security-performance tradeoffs in lattice cryptography. Create robust algorithms that intentionally add controlled noise for privacy or fault tolerance using structured error introduction.",
        "Polarization-adjusted convolutional codes pre-transform polar codes to improve distance properties for error correction. Create preprocessing stages that enhance algorithmic performance through code transformation and optimization.",
        "Threshold FHE with small partial shares enables distributed homomorphic computation with optimized share sizes. Create distributed processing algorithms with minimal communication overhead using secret sharing and homomorphic computation.",
        "CKKS bootstrapping optimizations advance bootstrapping approximate homomorphic encryption with reduced noise growth. Create algorithms that maintain precision through iterative operations using noise management techniques and error control.",
        "FHE hardware acceleration uses Intel AVX-512 based optimizations for polynomial multiplication and NTT operations. Create vectorized implementations for bulk data processing using SIMD-based approaches and polynomial arithmetic.",
        "Homomorphic encryption for AI integrates FHE with transformer models for private inference on encrypted data. Create algorithms that process sensitive data without exposure using encrypted computation patterns and privacy-preserving techniques.",
        "Circuit privacy in FHE hides computational structure during homomorphic evaluation for enhanced security. Create algorithms that obscure their internal operations for security using structure-hiding techniques and privacy protection.",
        "5G NR polar code enhancements update successive cancellation list decoding with CRC-aided error correction. Create dynamic data structure algorithms with self-correcting capabilities using adaptive list management and error detection.",
        "Spatially coupled LDPC codes advance quantum LDPC codes with improved threshold properties for quantum error correction. Create algorithms that leverage geometric relationships in data for better error resilience using spatial coupling techniques.",
        "Twin-field quantum key distribution error correction uses novel reconciliation protocols for quantum channel errors. Create algorithms designed for noisy, probabilistic environments using quantum error patterns and reconciliation methods.",
        "High-dimensional quantum error correction uses XYZ product codes for biased noise environments. Create algorithms that handle correlated failures across multiple dimensions using multi-dimensional error correction techniques.",
        "Polar code fast-adaptive SCL decoders implement variable throughput and latency for adaptive error correction. Create algorithms that dynamically adjust computational complexity based on data characteristics using adaptive decoding strategies.",
        "LMCompress breakthrough uses large language models for lossless compression achieving 2x improvement over traditional methods. Create algorithms that leverage learned patterns for data optimization using semantic understanding approaches.",
        "Entropy-aware block transforms use novel block-based linear transforms that outperform JPEG-LS and CALIC predictors. Create algorithms that reorganize data for optimal processing efficiency using entropy reduction techniques and block transformations.",
        "Divide-and-conquer genomic compression classifies subsequences by similarity before compression for enhanced efficiency. Create algorithms that group similar data elements for batch processing using classification-based approaches and similarity analysis.",
        "Context-based neural compression uses advanced prediction techniques with transformer models for entropy coding. Create algorithms that anticipate future data patterns for preemptive optimization using contextual prediction and neural techniques.",
        "Asymmetric numeral systems combine Huffman speed with arithmetic coding compression rates for optimal entropy coding. Create algorithms that balance speed and optimality based on data characteristics using asymmetric encoding approaches.",
        "Signal stability detection adaptive Kalman filter uses standard deviation assessment for adaptive noise parameter adjustment. Create algorithms that monitor their own performance and self-adjust parameters using stability detection techniques.",
        "KalmanFormer transformer-based approach models Kalman gains using attention mechanisms for enhanced filtering. Create algorithms that combine learned and analytical components using neural-classical hybrid approaches and attention-based modeling.",
        "Federated adaptive Kalman filtering enables multi-sensor fusion with dynamic reliability weighting and degradation isolation. Create algorithms that combine multiple unreliable data sources intelligently using sensor fusion approaches.",
        "Parallel adaptive Kalman filters use multiple filter approach for different signal components with cross-validation. Create algorithms that use multiple specialized processors for different data aspects using parallel processing techniques.",
        "Sage-Husa adaptive noise estimation performs real-time noise covariance matrix estimation with stability guarantees. Create algorithms that automatically adjust to changing environmental conditions using adaptive noise modeling techniques.",
        "zk-STARKs scalability improvements advance transparent, scalable zero-knowledge proofs without trusted setup. Create algorithms that provide inherent verifiability without external validation using transparency properties and scalable proof systems.",
        "Collaborative zk-SNARKs enable distributed secret proof generation across multiple parties with minimal overhead. Create algorithms that distribute sensitive computations while maintaining privacy using collaborative approaches and zero-knowledge techniques.",
        "Succinct polynomial commitments advance compact polynomial commitment schemes like Ligerito for verification. Create algorithms that compress complex mathematical structures for efficient verification using succinct representation techniques.",
        "Plonkish proof systems advance universal SNARKs with custom gates and lookup tables for general computation. Create algorithms that provide general-purpose computation frameworks with specialization options using universal construction techniques.",
        "Binius with M3 arithmetization optimizes binary field operations for zero-knowledge proofs. Create algorithms optimized for boolean operations and bit-level processing using binary field approaches and boolean optimization.",
        "Twin-field QKD protocols overcome rate-distance limits with square-root scaling instead of linear scaling. Create algorithms that maintain performance over extended network topologies using distance scaling breakthrough techniques.",
        "High-dimensional QKD uses 4-dimensional hybrid time-path encoding over deployed multicore fiber for enhanced capacity. Create algorithms that encode information across multiple parameters simultaneously using dimensional expansion techniques.",
        "Isogeny-based post-quantum signatures use SQIsign2D-West and SQIPrime developments with Deuring correspondence. Create algorithms based on graph transformations and mathematical equivalences using isogeny mathematics and algebraic structures.",

        # Biology & Systems (35 themes)
        "PeCHYRON prime editing systems use self-propagating DNA memory architecture that iteratively adds 3-nucleotide signatures while generating target sites. Create self-extending data structures that dynamically create new memory addresses as they grow with unlimited expansion capability.",
        "Solid-state DNA origami registers enable high-speed sequential DNA computing using origami-based registers with visual debugging capabilities. Create register-based computing architectures with built-in error visualization and correction mechanisms for transparent state management.",
        "Triplex-directed photo-cross-linking uses DNA origami stabilization through super-staples that weave structures together via light-activated cross-linking. Create self-reinforcing data structures that strengthen critical pathways based on usage patterns and light-activated stabilization.",
        "Willow chip surface code implementation achieves quantum error correction with logical error rates below physical rates. Create hierarchical error correction algorithms that improve accuracy through multi-level redundancy and exponential error suppression.",
        "Chemically conjugated branched staples use super-DNA origami technique with branched staples for creating robust complex nanostructures. Create branching algorithms that generate multiple execution paths with error resilience and enhanced structural stability.",
        "Orthogonal ribosome/mRNA pairs use engineered ribosomes with altered anti-Shine-Dalgarno sequences for programmable translation control. Create selective execution environments where different code segments run on specialized virtual machines with orthogonal instruction sets.",
        "Systematic sRNA design rules enhance efficiency and specificity of synthetic small RNAs through seed region length analysis. Create pattern matching algorithms with tunable specificity based on match length optimization and systematic design principles.",
        "Hypoxia biosensor circuits use engineered genetic circuits that respond to low-oxygen conditions with programmable sensitivity thresholds. Create adaptive algorithms that adjust behavior based on resource availability conditions and environmental stress responses.",
        "Integrase-enabled synthetic logic uses bacterial conjugation systems with integrases to create intercellular logic circuits. Create distributed computing protocols that use biological-style message passing with permanent state changes and intercellular communication.",
        "Minimal criterion novelty search retains variants meeting minimal activity thresholds while avoiding trivial solutions in synthetic biology. Create evolutionary algorithms that maintain diversity while ensuring functional viability and avoiding local optima.",
        "Differentiable logic cellular automata use neural cellular automata that learn transition rules using gradient descent while preserving discrete nature. Create learning algorithms that evolve their own update rules through backpropagation while maintaining discrete computational properties.",
        "Reversible universal pattern generators use one-dimensional cellular automata with finite initial configurations generating all finite patterns. Create compression algorithms that can reconstruct any pattern from minimal initial conditions using universal generation rules.",
        "Multi-scale fractal generation uses cellular automata rules that produce self-similar patterns at different scales like Sierpiński triangles. Create hierarchical algorithms that exhibit consistent behavior across different levels of abstraction with fractal properties.",
        "Maze-generating cellular automata use rules like B3/S12345 that evolve random starting patterns into complex maze structures. Create procedural generation algorithms for creating navigable spaces with complex topological properties and emergent structure.",
        "Paranemic crossover pattern generation constructs complex molecular topology using DNA parallel crossover motifs with programmable crossing numbers. Create graph algorithms that generate controlled topological complexity and knotted structures with precise geometric properties.",
        "Thousand-robot self-assembly uses Kilobot-style systems achieving complex preprogrammed shapes through local communication within three body diameters. Create distributed algorithms that achieve global objectives using only local neighborhood information and emergent coordination.",
        "Pogobot vibration-based locomotion uses swarm robotics platform with vibration-based movement and infrared communication with directional messaging. Create distributed systems that use simple physical mechanisms for complex coordination and emergent collective behavior.",
        "Acoustic swarm navigation achieves centimeter-accuracy robot swarms that navigate cooperatively using only sound signals without cameras. Create location-based algorithms that achieve precision through acoustic triangulation and cooperative positioning techniques.",
        "Stigmergy-based coordination uses indirect communication through environmental modification allowing robots to coordinate complex tasks through environmental traces. Create cache-based algorithms that use persistent environmental state for coordination and indirect communication mechanisms.",
        "Formation-flying fault tolerance uses swarm algorithms that handle outliers and failed units while maintaining collective behavior. Create resilient distributed systems that maintain functionality despite node failures and dynamic membership changes.",
        "Diacetylene-based photosensitive templates replace DNA sugar-phosphate backbone with photosensitive diacetylene for UV-triggered polymerization. Create light-activated computing systems that change behavior based on optical input and photochemical state transitions.",
        "Two-helix DNA origami conductance shows identical helices exhibiting different conductance based on electrode positioning. Create adaptive circuit designs that change electrical properties based on connection points and electrode configuration.",
        "Gold-tellurium nanojunctions create electrically linked metal-semiconductor junctions through location-specific binding on DNA origami scaffolds. Create hybrid computing architectures that combine metallic and semiconductor properties with precise spatial control.",
        "DNA wireframe electrode assembly positions enzyme pairs at sub-10nm resolution within DNA nanotubes for enhanced cascade efficiency. Create tightly coupled processing pipelines with nanoscale precision and enhanced reaction efficiency.",
        "Conductive nanowire assembly uses DNA origami-directed assembly of gold nanorods into 130nm conductive wires with measured resistance. Create self-assembling circuit elements that form conductive pathways on demand with programmable electrical properties.",
        "Liquid crystal active materials respond to stimuli by changing shape, color, or properties in controlled programmable ways. Create data structures that physically reconfigure based on access patterns or computational needs with responsive material properties.",
        "Modular robot self-reconfiguration uses algorithms that prevent docking positions from becoming unreachable during large-scale assembly. Create memory management systems that prevent fragmentation during dynamic allocation with accessibility preservation.",
        "Sequence-dependent assembly planning uses distributed algorithms that generate attraction lists for available positions avoiding impossible configurations. Create scheduling algorithms that consider future accessibility when making current decisions with forward-looking optimization.",
        "Freeze-dried cellulose aerogels combine cellulose nanofibers with metallic phases for enhanced fire resistance and thermal properties. Create layered security systems that use multiple materials for protection with hierarchical defense mechanisms.",
        "Graphene mesosponge architecture uses 3D carbon nanomaterials with controlled mesoporous frameworks providing large surface areas. Create network topologies that maximize connectivity while maintaining structural integrity and optimal surface utilization.",
        "Competitive swarm optimizer with mutated agents uses competitive selection combined with mutation for complex statistical optimization. Create hybrid algorithms that combine competition and variation for improved solution quality and robust optimization.",
        "Jellyfish search optimizer uses bio-inspired metaheuristic based on jellyfish food-finding behavior in ocean currents. Create search algorithms that use current-following behavior for exploration and adaptive navigation in solution spaces.",
        "Bobcat optimization algorithm mimics bobcat hunting behavior for supply chain optimization with prey stalking and territorial marking. Create resource allocation algorithms that use territory-based optimization and hunting-inspired search strategies.",
        "Electric eel foraging optimization uses electric eel hunting patterns with electrical field generation for prey detection and capture. Create sensor-based search algorithms that use field-based detection mechanisms and electrical sensing techniques.",
        "Multi-omics integration networks use computational approaches combining genomics, proteomics, and metabolomics data for comprehensive biological analysis. Create multi-source data fusion algorithms that integrate heterogeneous information streams with biological network principles."
    ]




    tasks = [
    # --- Perspective: Core Performance & Instruction Set ---
    "Develop an x86-64 assembly function that outperforms the standard C `memcpy`, minimizing the total number of executed low-level instructions.",
    "Write an x86 assembly implementation of `memcpy` that leverages SIMD instructions (SSE, AVX) to process larger blocks of memory per instruction cycle.",
    "Construct a `memcpy` routine in x86-64 assembly that prioritizes reducing instruction latency and maximizing throughput on modern Intel/AMD microarchitectures.",
    "Design a `memcpy` alternative in pure x86-64 assembly using `rep movsb` and analyze its performance against a SIMD-based approach for various data sizes.",
    "Implement `memcpy` in x86-64 assembly, focusing on minimizing micro-op fusion and optimizing the instruction decoding pipeline.",
    "Create an x86-64 assembly `memcpy` that dynamically selects the optimal copy strategy (e.g., byte-wise, SIMD, `rep movs`) based on the input size.",
    "Write an assembly function for x86-64 that serves as a faster `memcpy`, specifically by avoiding branch mispredictions for size and alignment checks.",
    "Develop a `memcpy` implementation in x86-64 assembly that utilizes non-temporal store instructions (e.g., `movntdq`) to minimize cache pollution for large data transfers.",
    "Generate an x86-64 assembly version of `memcpy` that uses software prefetching (`prefetchnta`, `prefetcht0/1/2`) to hide memory latency.",
    "Construct an x86 assembly `memcpy` that is optimized for small, unaligned memory copies by minimizing the overhead of setup and alignment handling.",
    "Build a `memcpy` in x86-64 assembly that leverages AVX-512 instructions for maximum parallelism on compatible processors, ensuring a significant reduction in total instructions.",
    "Design a `memcpy` function in x86 assembly that focuses on minimizing register dependencies to improve out-of-order execution performance.",
    "Implement a `memcpy` in x86-64 assembly specifically for a target microarchitecture (e.g., Zen 4 or Golden Cove), exploiting its unique instruction scheduling and port capabilities.",
    "Write a `memcpy` variant in x86-64 assembly that uses overlapping reads and writes with SIMD registers to hide memory read latency.",
    "Create an x86 assembly `memcpy` that relies exclusively on general-purpose registers and integer instructions, avoiding the FPU/SIMD state-switching overhead.",
    "Develop a `memcpy` in x86-64 assembly that employs loop unrolling with vector instructions to reduce loop overhead and increase instruction-level parallelism.",
    "Generate a `memcpy` in x86-64 assembly optimized to reduce read-for-ownership (RFO) stalls by using specialized store instructions.",
    "Write an x86-64 assembly function that replicates `memcpy`'s functionality with fewer instructions by exploiting the Enhanced REP MOVSB (ERMSB) feature.",
    "Construct a `memcpy` in x86-64 assembly that is optimized for copying memory regions that are known to be 64-byte aligned.",
    "Design an x86 assembly `memcpy` that minimizes pipeline stalls by carefully scheduling memory load and store operations.",

    # --- Perspective: Cache & Memory Hierarchy ---
    "Create a `memcpy` function in x86-64 assembly that is optimized for L1 data cache throughput, ensuring source and destination data streams are managed efficiently.",
    "Write an x86-64 assembly `memcpy` designed to minimize cache line splits by handling unaligned source and destination addresses intelligently.",
    "Develop a `memcpy` in x86-64 assembly that strategically uses non-temporal hints to bypass the cache hierarchy for very large copy operations.",
    "Construct a `memcpy` routine in x86 assembly that aims to saturate memory bus bandwidth, focusing on instruction choice over raw instruction count.",
    "Generate an x86-64 assembly `memcpy` that is 'cache-aware', prefetching data just in time to be loaded into L1d from L2/L3 cache.",
    "Implement a `memcpy` in x86-64 assembly that minimizes partial writes to cache lines, thereby reducing memory controller traffic.",
    "Design a faster `memcpy` in x86 assembly that avoids self-eviction from the cache when source and destination addresses are close to each other.",
    "Write an x86-64 assembly `memcpy` that explicitly manages data alignment to ensure all vector loads and stores are on optimal memory boundaries.",
    "Develop a `memcpy` function in assembly that is optimized for systems with a specific cache line size (e.g., 64 bytes), reducing redundant memory fetches.",
    "Create an x86-64 assembly `memcpy` that mitigates the performance penalty of crossing page boundaries during a copy operation.",
    "Construct a `memcpy` in x86-64 assembly that is sensitive to the Non-Uniform Memory Access (NUMA) architecture, optimizing for local node copies.",
    "Build a `memcpy` using x86 assembly that prioritizes keeping the translation lookaside buffer (TLB) hot by optimizing memory access patterns.",
    "Design a `memcpy` variant in x86-64 assembly that uses write-combining techniques to buffer small writes into a larger, more efficient burst.",
    "Write an assembly `memcpy` for x86-64 that reduces the dependency on the store buffer by using instructions that bypass it where appropriate.",
    "Implement a `memcpy` function in x86-64 assembly specifically for copying to memory-mapped I/O (MMIO) regions, using fence instructions where necessary.",

    # --- Perspective: System-Level & Edge Cases ---
    "As a systems programmer, write a `memcpy` in x86-64 assembly that is faster than glibc's version by reducing prologue and epilogue overhead for common data sizes.",
    "Your task is to create a `memcpy` function entirely in x86-64 assembly that minimizes its instruction footprint for use in a bootloader or embedded environment.",
    "Generate an x86 assembly `memcpy` that handles overlapping source and destination regions correctly (like `memmove`) but does so with fewer instructions than a naive check-and-branch implementation.",
    "Design a `memcpy` in x86-64 assembly that is robust against edge cases, such as zero-length copies and heavily unaligned pointers, while still minimizing instruction count.",
    "Write an x86 assembly `memcpy` that is optimized for a scenario where the source data is 'hot' (in cache) and the destination is 'cold' (not in cache).",
    "Develop a `memcpy` in x86-64 assembly that is optimized for the reverse scenario: a 'cold' source and a 'hot' destination.",
    "Create a `memcpy` implementation in x86-64 assembly that returns profiling data, such as the number of alignment-handling branches taken, without significantly impacting performance.",
    "Construct a `memcpy` in x86 assembly that forgoes AVX to maintain compatibility with older x86-64 CPUs but still beats a standard C implementation through clever instruction scheduling.",
    "Build an x86-64 assembly `memcpy` that is position-independent (PIC) and suitable for use in shared libraries, while minimizing instruction overhead.",
    "Design a `memcpy` function in pure x86-64 assembly intended for a real-time operating system (RTOS), where predictable execution time is more critical than raw throughput.",
    "Write a `memcpy` in x86-64 assembly that dynamically adapts its strategy based on runtime CPU feature detection (e.g., checking for AVX2, AVX-512 support).",
    "Implement a `memcpy` in x86 assembly that focuses on reducing context switching overhead by avoiding instructions that require saving/restoring large vector registers.",
    "Create a version of `memcpy` in x86-64 assembly that is optimized for copying sparse data, skipping over blocks of zeros with minimal instructions.",
    "Develop a `memcpy` in x86-64 assembly designed for a high-frequency trading (HFT) environment, where minimizing latency for small, fixed-size packets is the primary goal.",
    "Generate an x86 assembly `memcpy` that is optimized for power efficiency, selecting instructions that consume less energy on modern cores.",

    # --- Perspective: Abstract & Conceptual ---
    "Reimagine `memcpy` in x86-64 assembly, focusing on an instruction stream that maximizes the CPU's ability to perform memory disambiguation.",
    "Code a `memcpy` function in x86-64 assembly where the primary goal is to reduce the number of cycles the pipeline is stalled waiting for memory operations.",
    "Create a `memcpy` implementation in x86 assembly that minimizes data-dependent control flow, producing a more predictable instruction stream.",
    "Your objective is to write an x86-64 assembly `memcpy` that treats memory copy as a data streaming problem, optimizing for continuous, uninterrupted data flow.",
    "Design a `memcpy` in x86 assembly that minimizes resource conflicts on the CPU's execution ports, particularly the AGUs (Address Generation Units) and memory ports.",
    "Implement `memcpy` in x86-64 assembly with a focus on maximizing instruction-level parallelism (ILP) for the entire function.",
    "Write a `memcpy` in x86-64 assembly that achieves higher speed by reducing the total number of memory accesses, even if it means slightly more ALU instructions.",
    "Construct a `memcpy` function in x86-64 assembly from first principles, assuming no standard library is available and aiming for the lowest possible instruction count.",
    "Develop an x86-64 assembly `memcpy` that is 'branchless' for its main copy loop, handling alignment and tail-end copies outside the core transfer logic.",
    "Generate a `memcpy` in x86 assembly that performs a 'dual-stream' copy, using separate prefetch streams for the source and destination to improve memory parallelism.",
    "Code an x86-64 assembly `memcpy` that prioritizes minimizing the time-to-first-byte copied, even if the overall throughput for large copies is not the absolute maximum.",
    "Design a `memcpy` in x86-64 assembly whose instruction sequence is optimized for micro-op decomposition and retirement rate on out-of-order processors.",
    "Create a `memcpy` in x86 assembly that uses advanced bit manipulation techniques to handle unaligned start and end segments with a minimal number of instructions.",
    "Implement `memcpy` in x86-64 assembly to function as a high-performance DMA (Direct Memory Access) engine emulator in software, using non-temporal stores.",
    "Write a `memcpy` in x86 assembly that exploits temporal locality for medium-sized copies that are likely to be re-read soon, avoiding cache-bypassing stores.",
    "Develop a `memcpy` function in x86-64 assembly that is optimized to reduce the pressure on the register allocation file by reusing registers efficiently.",
    "Construct an x86-64 assembly `memcpy` that aims to complete in the fewest possible clock cycles on average, considering a typical distribution of copy sizes.",
    "Design a `memcpy` in x86 assembly where the primary optimization goal is to reduce the dependency chain length of the instructions in the main loop.",
    "Create a `memcpy` in x86-64 assembly that is optimized for multi-threaded scenarios, minimizing cache coherence traffic by using appropriate memory ordering semantics.",
    "Write an x86-64 assembly version of `memcpy` that is algorithmically superior, using a multi-level strategy (e.g., small, medium, large copy handlers) with minimal dispatch overhead.",
    "Implement a `memcpy` in x86 assembly with a focus on minimizing the code size (icache footprint) of the function itself without sacrificing significant performance.",
    "Generate a `memcpy` in x86-64 assembly that is tailored for copying data between different memory types, such as from WC (Write-Combining) to WB (Write-Back) memory.",
    "Your goal is a `memcpy` in x86-64 assembly that minimizes the number of retired micro-ops, a key metric for performance on modern CPUs."
]
