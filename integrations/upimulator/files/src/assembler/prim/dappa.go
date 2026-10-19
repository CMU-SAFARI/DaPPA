package prim

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

// --- Recipe schema (version 1) ----------------------------------------------

type recipePipeline struct {
	DataLength       int64 `json:"data_length"`
	ElementSizeBytes int   `json:"element_size_bytes"`
	DpuCount         int   `json:"dpu_count"`
	NumTasklets      int   `json:"num_tasklets"`
	NumExecutions    int   `json:"num_executions"`
	CacheSizeBytes   int   `json:"cache_size_bytes"`
}

type recipeArg struct {
	Index               int      `json:"index"`
	Type                string   `json:"type"`
	ElementSizeBytes    int      `json:"element_size_bytes"`
	Flags               []string `json:"flags"`
	MramOffset          int64    `json:"mram_offset"`
	LengthElements      int64    `json:"length_elements"`
	TransferOverlap     int      `json:"transfer_overlap"`
	MramOverlapElements int      `json:"mram_overlap_elements"`
	FilterLengthSymbol  string   `json:"filter_length_symbol,omitempty"`
	WramSymbol          string   `json:"wram_symbol,omitempty"`
}

type recipeHostData struct {
	TaskletElementCount         uint32 `json:"tasklet_element_count"`
	AdditionalElementTaskletCnt uint32 `json:"additional_element_tasklet_count"`
}

type recipeMramInput struct {
	ArgIndex int      `json:"arg_index"`
	DpuFiles []string `json:"dpu_files"`
}

type recipeWramInput struct {
	ArgIndex int    `json:"arg_index"`
	ValueBin string `json:"value_bin"`
	PerDpu   bool   `json:"per_dpu"`
}

type recipeMramOutput struct {
	ArgIndex int `json:"arg_index"`
}

// recipeLookup is a LOOKUP_INPUT arg. Its array is the same on every DPU and in
// every round, so the recipe stores it once and it is copied into each DPU's MRAM
// heap at MramOffset.
type recipeLookup struct {
	ArgIndex         int    `json:"arg_index"`
	MramOffset       int64  `json:"mram_offset"`
	LengthElements   int64  `json:"length_elements"`
	ElementSizeBytes int    `json:"element_size_bytes"`
	ValueBin         string `json:"value_bin"`
}

type recipeExecution struct {
	Index                 int            `json:"index"`
	ElementsPerDpu        int            `json:"elements_per_dpu"`
	BaseArgOffsetElements int64          `json:"base_arg_offset_elements"`
	HostData              recipeHostData `json:"host_data"`
	Inputs                struct {
		Mram []recipeMramInput `json:"mram"`
		Wram []recipeWramInput `json:"wram"`
	} `json:"inputs"`
	Outputs struct {
		Mram        []recipeMramOutput `json:"mram"`
		WramSymbols []string           `json:"wram_symbols"`
	} `json:"outputs"`
}

type recipe struct {
	SchemaVersion int               `json:"$schema_version"`
	DeviceSource  string            `json:"device_source"`
	Pipeline      recipePipeline    `json:"pipeline"`
	Args          []recipeArg       `json:"args"`
	Lookups       []recipeLookup    `json:"lookups"`
	Executions    []recipeExecution `json:"executions"`
	// host_post_processing is only used by the end-to-end cost model.
}

// --- Assemblable implementation ---------------------------------------------

// Dappa is a generic Assemblable that replays any DaPPA pipeline from the recipe
// written by Pipeline::execute() when DAPPA_EMIT_RECIPE is set. The recipe holds
// the MRAM offset of every arg, the per-round host data and the per-DPU input
// files, so no workload-specific Go code is needed. The recipe directory is read
// from DAPPA_RECIPE_DIR, since data_prep_params only accepts integers.
type Dappa struct {
	recipe      recipe
	recipeDir   string
	argByIndex  map[int]*recipeArg
	mramBufSize int64 // Covers [offset, offset+bytes) of every arg
}

func (this *Dappa) Init(parser *misc.CommandLineParser) {
	dir := os.Getenv("DAPPA_RECIPE_DIR")
	if dir == "" {
		panic(errors.New("DAPPA_RECIPE_DIR is not set; export it to the directory that holds recipe.json"))
	}
	this.recipeDir = dir

	body, err := os.ReadFile(filepath.Join(dir, "recipe.json"))
	if err != nil {
		panic(fmt.Errorf("read recipe.json: %w", err))
	}
	if err := json.Unmarshal(body, &this.recipe); err != nil {
		panic(fmt.Errorf("parse recipe.json: %w", err))
	}
	if this.recipe.SchemaVersion != 1 {
		panic(fmt.Errorf("recipe schema version %d unsupported (expected 1)", this.recipe.SchemaVersion))
	}

	// The per-DPU input files only exist for the DPU count the recipe was emitted for.
	cliDpus := int(parser.IntParameter("num_channels")) *
		int(parser.IntParameter("num_ranks_per_channel")) *
		int(parser.IntParameter("num_dpus_per_rank"))
	if cliDpus != this.recipe.Pipeline.DpuCount {
		panic(fmt.Errorf(
			"DPU count mismatch: simulator configured for %d, recipe was emitted for %d. "+
				"Set --num_channels/--num_ranks_per_channel/--num_dpus_per_rank so their product equals %d.",
			cliDpus, this.recipe.Pipeline.DpuCount, this.recipe.Pipeline.DpuCount))
	}

	this.argByIndex = make(map[int]*recipeArg, len(this.recipe.Args))
	for i := range this.recipe.Args {
		this.argByIndex[this.recipe.Args[i].Index] = &this.recipe.Args[i]
	}

	// Size the per-DPU MRAM buffer from each arg's offset plus its largest per-DPU
	// slice. LengthElements is the total over all DPUs, so using it directly would
	// overflow the 64 MB MRAM.
	for _, exec := range this.recipe.Executions {
		perDpu := int64(exec.ElementsPerDpu)
		for _, m := range exec.Inputs.Mram {
			arg := this.argByIndex[m.ArgIndex]
			// TransferOverlap covers the extra elements read by WINDOW stages.
			end := arg.MramOffset + (perDpu+int64(arg.TransferOverlap))*int64(arg.ElementSizeBytes)
			if end > this.mramBufSize {
				this.mramBufSize = end
			}
		}
		for _, m := range exec.Outputs.Mram {
			arg := this.argByIndex[m.ArgIndex]
			// GROUP and REDUCE stages produce fewer elements than they consume, so
			// size outputs from their own length rather than from the input slice.
			outPerDpu := perDpu
			if dpuCount := int64(this.recipe.Pipeline.DpuCount); dpuCount > 0 {
				outPerDpu = (arg.LengthElements + dpuCount - 1) / dpuCount
			}
			end := arg.MramOffset + outPerDpu*int64(arg.ElementSizeBytes)
			if end > this.mramBufSize {
				this.mramBufSize = end
			}
		}
	}

	// Lookup arrays occupy a fixed region for the whole run.
	for _, lk := range this.recipe.Lookups {
		end := lk.MramOffset + lk.LengthElements*int64(lk.ElementSizeBytes)
		if end > this.mramBufSize {
			this.mramBufSize = end
		}
	}
}

func (this *Dappa) NumExecutions() int {
	return len(this.recipe.Executions)
}

// InputDpuMramHeapPointerName returns the MRAM heap contents at launch: every input
// arg's bytes at its MRAM offset. Intermediate args are left zero, since a stage
// always writes them before a later stage reads them.
func (this *Dappa) InputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.checkBounds(execution, dpuID)
	exec := this.recipe.Executions[execution]

	buf := make([]byte, this.mramBufSize)
	for _, m := range exec.Inputs.Mram {
		arg, ok := this.argByIndex[m.ArgIndex]
		if !ok {
			panic(fmt.Errorf("execution %d references unknown arg index %d", execution, m.ArgIndex))
		}
		if dpuID >= len(m.DpuFiles) {
			panic(fmt.Errorf("arg %d has no dpu_files entry for DPU %d in execution %d",
				m.ArgIndex, dpuID, execution))
		}
		bin, err := os.ReadFile(filepath.Join(this.recipeDir, m.DpuFiles[dpuID]))
		if err != nil {
			panic(fmt.Errorf("read %s: %w", m.DpuFiles[dpuID], err))
		}
		off := arg.MramOffset
		if off+int64(len(bin)) > int64(len(buf)) {
			panic(fmt.Errorf("arg %d bin (%d bytes at offset %d) exceeds MRAM buffer (%d)",
				m.ArgIndex, len(bin), off, len(buf)))
		}
		copy(buf[off:], bin)
	}

	// Lookup arrays are the same for every execution and DPU.
	for _, lk := range this.recipe.Lookups {
		bin, err := os.ReadFile(filepath.Join(this.recipeDir, lk.ValueBin))
		if err != nil {
			panic(fmt.Errorf("read lookup %s: %w", lk.ValueBin, err))
		}
		off := lk.MramOffset
		if off+int64(len(bin)) > int64(len(buf)) {
			panic(fmt.Errorf("lookup arg %d bin (%d bytes at offset %d) exceeds MRAM buffer (%d)",
				lk.ArgIndex, len(bin), off, len(buf)))
		}
		copy(buf[off:], bin)
	}
	return 0, bytesToStream(buf)
}

// OutputDpuMramHeapPointerName returns the reference MRAM contents after launch.
// Recipe emission never runs the pipeline, so there is no reference output; the
// buffer is zero-filled and verification must be disabled with DAPPA_SKIP_VERIFY=1.
func (this *Dappa) OutputDpuMramHeapPointerName(execution int, dpuID int) (int64, *encoding.ByteStream) {
	this.checkBounds(execution, dpuID)
	buf := make([]byte, this.mramBufSize)
	return 0, bytesToStream(buf)
}

// InputDpuHost returns the contents of each __host WRAM symbol at launch:
// hostData (taskletElementCount and additionalElementTaskletCount, two uint32)
// and the scalar args argScalarRaw{N}, whose bytes are read from value_bin.
func (this *Dappa) InputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
	this.checkBounds(execution, dpuID)
	exec := this.recipe.Executions[execution]

	out := make(map[string]*encoding.ByteStream)

	// struct HostData from include/shared.h
	hd := newStream()
	appendUint32(hd, exec.HostData.TaskletElementCount)
	appendUint32(hd, exec.HostData.AdditionalElementTaskletCnt)
	out["hostData"] = hd

	// value_bin already contains every WRAM copy of the scalar
	for _, w := range exec.Inputs.Wram {
		arg, ok := this.argByIndex[w.ArgIndex]
		if !ok {
			panic(fmt.Errorf("execution %d wram input references unknown arg index %d",
				execution, w.ArgIndex))
		}
		if arg.WramSymbol == "" {
			panic(fmt.Errorf("arg %d has wram input but no wram_symbol declared", w.ArgIndex))
		}
		bin, err := os.ReadFile(filepath.Join(this.recipeDir, w.ValueBin))
		if err != nil {
			panic(fmt.Errorf("read %s: %w", w.ValueBin, err))
		}
		out[arg.WramSymbol] = bytesToStream(bin)
	}

	return out
}

// OutputDpuHost returns zero-filled reference values for the __host WRAM symbols
// read back after launch (see OutputDpuMramHeapPointerName).
func (this *Dappa) OutputDpuHost(execution int, dpuID int) map[string]*encoding.ByteStream {
	this.checkBounds(execution, dpuID)
	exec := this.recipe.Executions[execution]

	out := make(map[string]*encoding.ByteStream)
	for _, sym := range exec.Outputs.WramSymbols {
		bs := newStream()
		// WRAM outputs are uint32_t[NR_TASKLETS] (e.g., filterOutputLengthArg{N})
		for t := 0; t < this.recipe.Pipeline.NumTasklets; t++ {
			appendUint32(bs, 0)
		}
		out[sym] = bs
	}
	return out
}

// --- helpers ----------------------------------------------------------------

func (this *Dappa) checkBounds(execution, dpuID int) {
	if execution < 0 || execution >= len(this.recipe.Executions) {
		panic(fmt.Errorf("execution %d out of range [0,%d)", execution, len(this.recipe.Executions)))
	}
	if dpuID < 0 || dpuID >= this.recipe.Pipeline.DpuCount {
		panic(fmt.Errorf("dpu_id %d out of range [0,%d)", dpuID, this.recipe.Pipeline.DpuCount))
	}
}

func newStream() *encoding.ByteStream {
	bs := new(encoding.ByteStream)
	bs.Init()
	return bs
}

func appendUint32(bs *encoding.ByteStream, v uint32) {
	w := new(word.Word)
	w.Init(32)
	w.SetValue(int64(v))
	bs.Merge(w.ToByteStream())
}

func bytesToStream(b []byte) *encoding.ByteStream {
	bs := newStream()
	for _, x := range b {
		bs.Append(uint8(x))
	}
	return bs
}
