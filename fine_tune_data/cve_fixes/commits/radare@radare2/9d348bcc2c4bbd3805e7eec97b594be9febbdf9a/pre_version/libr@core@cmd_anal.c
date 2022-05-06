/* radare - LGPL - Copyright 2009-2018 - pancake, maijin */

#include "r_util.h"
#include "r_core.h"

static const char *help_msg_a[] = {
	"Usage:", "a", "[abdefFghoprxstc] [...]",
	"aa", "[?]", "analyze all (fcns + bbs) (aa0 to avoid sub renaming)",
	"a8", " [hexpairs]", "analyze bytes",
	"ab", "[b] [addr]", "analyze block at given address",
	"abb", " [len]", "analyze N basic blocks in [len] (section.size by default)",
	"ac", " [cycles]", "analyze which op could be executed in [cycles]",
	"ad", "[?]", "analyze data trampoline (wip)",
	"ad", " [from] [to]", "analyze data pointers to (from-to)",
	"ae", "[?] [expr]", "analyze opcode eval expression (see ao)",
	"af", "[?]", "analyze Functions",
	"aF", "", "same as above, but using anal.depth=1",
	"ag", "[?] [options]", "output Graphviz code",
	"ah", "[?]", "analysis hints (force opcode size, ...)",
	"ai", " [addr]", "address information (show perms, stack, heap, ...)",
	"an"," [name] [@addr]","show/rename/create whatever flag/function is used at addr",
	"ao", "[?] [len]", "analyze Opcodes (or emulate it)",
	"aO", "[?] [len]", "Analyze N instructions in M bytes",
	"ap", "", "find prelude for current offset",
	"ar", "[?]", "like 'dr' but for the esil vm. (registers)",
	"as", "[?] [num]", "analyze syscall using dbg.reg",
	"av", "[?] [.]", "show vtables",
	"ax", "[?]", "manage refs/xrefs (see also afx?)",
	NULL
};

static const char *help_msg_aa[] = {
	"Usage:", "aa[0*?]", " # see also 'af' and 'afna'",
	"aa", " ", "alias for 'af@@ sym.*;af@entry0;afva'", //;.afna @@ fcn.*'",
	"aa*", "", "analyze all flags starting with sym. (af @@ sym.*)",
	"aaa", "[?]", "autoname functions after aa (see afna)",
	"aab", "", "aab across io.sections.text",
	"aac", " [len]", "analyze function calls (af @@ `pi len~call[1]`)",
	"aac*", " [len]", "flag function calls without performing a complete analysis",
	"aad", " [len]", "analyze data references to code",
	"aae", " [len] ([addr])", "analyze references with ESIL (optionally to address)",
	"aaE", "", "run aef on all functions (same as aef @@f)",
	"aaf", " ", "analyze all functions (e anal.hasnext=1;afr @@c:isq)",
	"aai", "[j]", "show info of all analysis parameters",
	"aan", "", "autoname functions that either start with fcn.* or sym.func.*",
	"aap", "", "find and analyze function preludes",
	"aar", "[?] [len]", "analyze len bytes of instructions for references",
	"aas", " [len]", "analyze symbols (af @@= `isq~[0]`)",
	"aat", " [len]", "analyze all consecutive functions in section",
	"aaT", " [len]", "analyze code after trap-sleds",
	"aau", " [len]", "list mem areas (larger than len bytes) not covered by functions",
	"aav", " [sat]", "find values referencing a specific section or map",
	NULL
};

static const char *help_msg_aar[] = {
	"Usage:", "aar", "[j*] [sz] # search and analyze xrefs",
	"aar", " [sz]", "analyze xrefs in current section or sz bytes of code",
	"aar*", " [sz]", "list found xrefs in radare commands format",
	"aarj", " [sz]", "list found xrefs in JSON format",
	NULL
};

static const char *help_msg_ab[] = {
	"Usage:", "ab", "",
	"ab", " [addr]", "show basic block information at given address",
	"abb", " [length]", "analyze N bytes and extract basic blocks",
	"abj", "", "display basic block information in JSON",
	"abx", " [hexpair-bytes]", "analyze N bytes",
	NULL
};

static const char *help_msg_ad[] = {
	"Usage:", "ad", "[kt] [...]",
	"ad", " [N] [D]", "analyze N data words at D depth",
	"ad4", " [N] [D]", "analyze N data words at D depth (asm.bits=32)",
	"ad8", " [N] [D]", "analyze N data words at D depth (asm.bits=64)",
	"adf", "", "analyze data in function (use like .adf @@=`afl~[0]`",
	"adfg", "", "analyze data in function gaps",
	"adt", "", "analyze data trampolines (wip)",
	"adk", "", "analyze data kind (code, text, data, invalid, ...)",
	NULL
};

static const char *help_msg_ae[] = {
	"Usage:", "ae[idesr?] [arg]", "ESIL code emulation",
	"ae", " [expr]", "evaluate ESIL expression",
	"ae?", "", "show this help",
	"ae??", "", "show ESIL help",
	"ae[aA]", "[f] [count]", "analyse esil accesses (regs, mem..)",
	"aec", "[?]", "continue until ^C",
	"aecs", " [sn]", "continue until syscall number",
	"aecu", " [addr]", "continue until address",
	"aecue", " [esil]", "continue until esil expression match",
	"aef", " [addr]", "emulate function",
	"aei", "", "initialize ESIL VM state (aei- to deinitialize)",
	"aeim", " [addr] [size] [name]", "initialize ESIL VM stack (aeim- remove)",
	"aeip", "", "initialize ESIL program counter to curseek",
	"aek", " [query]", "perform sdb query on ESIL.info",
	"aek-", "", "resets the ESIL.info sdb instance",
	"aep", "[?] [addr]", "manage esil pin hooks",
	"aepc", " [addr]", "change esil PC to this address",
	"aer", " [..]", "handle ESIL registers like 'ar' or 'dr' does",
	"aets", "[?]", "ESIL Trace session",
	"aes", "", "perform emulated debugger step",
	"aesp", " [X] [N]", "evaluate N instr from offset X",
	"aesb", "", "step back",
	"aeso", " ", "step over",
	"aesu", " [addr]", "step until given address",
	"aesue", " [esil]", "step until esil expression match",
	"aetr", "[esil]", "Convert an ESIL Expression to REIL",
	"aex", " [hex]", "evaluate opcode expression",
	NULL
};

static const char *help_detail_ae[] = {
	"Examples:", "ESIL", " examples and documentation",
	"+", "=", "A+=B => B,A,+=",
	"+", "", "A=A+B => B,A,+,A,=",
	"++", "", "increment, 2,A,++ == 3 (see rsi,--=[1], ... )",
	"--", "", "decrement, 2,A,-- == 1",
	"*", "=", "A*=B => B,A,*=",
	"/", "=", "A/=B => B,A,/=",
	"%", "=", "A%=B => B,A,%=",
	"&", "=", "and ax, bx => bx,ax,&=",
	"|", "", "or r0, r1, r2 => r2,r1,|,r0,=",
	"!", "=", "negate all bits",
	"^", "=", "xor ax, bx => bx,ax,^=",
	"", "[]", "mov eax,[eax] => eax,[],eax,=",
	"=", "[]", "mov [eax+3], 1 => 1,3,eax,+,=[]",
	"=", "[1]", "mov byte[eax],1 => 1,eax,=[1]",
	"=", "[8]", "mov [rax],1 => 1,rax,=[8]",
	"[]", "", "peek from random position",
	"[*]", "", "peek some from random position",
	"=", "[*]", "poke some at random position",
	"$", "", "int 0x80 => 0x80,$",
	"$$", "", "simulate a hardware trap",
	"==", "", "pops twice, compare and update esil flags",
	"<", "", "compare for smaller",
	"<", "=", "compare for smaller or equal",
	">", "", "compare for bigger",
	">", "=", "compare bigger for or equal",
	">>", "=", "shr ax, bx => bx,ax,>>=  # shift right",
	"<<", "=", "shl ax, bx => bx,ax,<<=  # shift left",
	">>>", "=", "ror ax, bx => bx,ax,>>>=  # rotate right",
	"<<<", "=", "rol ax, bx => bx,ax,<<<=  # rotate left",
	"?{", "", "if popped value != 0 run the block until }",
	"POP", "", "drops last element in the esil stack",
	"DUP", "", "duplicate last value in stack",
	"NUM", "", "evaluate last item in stack to number",
	"PICK", "", "pick Nth element in stack",
	"RPICK", "", "pick Nth element in reversed stack",
	"SWAP", "", "swap last two values in stack",
	"TRAP", "", "stop execution",
	"BITS", "", "16,BITS  # change bits, useful for arm/thumb",
	"TODO", "", "the instruction is not yet esilized",
	"STACK", "", "show contents of stack",
	"CLEAR", "", "clears the esil stack",
	"REPEAT", "", "repeat n times",
	"BREAK", "", "terminates the string parsing",
	"GOTO", "", "jump to the Nth word popped from the stack",
	NULL
};

static const char *help_msg_aea[] = {
	"Examples:", "aea", " show regs used in a range",
	"aea", " [ops]", "Show regs used in N instructions (all,read,{no,}written,memreads,memwrites)",
	"aea*", " [ops]", "Create mem.* flags for memory accesses",
	"aeaf", "", "Show regs used in current function",
	"aear", " [ops]", "Show regs read in N instructions",
	"aeaw", " [ops]", "Show regs written in N instructions",
	"aean", " [ops]", "Show regs not written in N instructions",
	"aeaj", " [ops]", "Show aea output in JSON format",
	"aeA", " [len]", "Show regs used in N bytes (subcommands are the same)",
	NULL
};

static const char *help_msg_aec[] = {
	"Examples:", "aec", " continue until ^c",
	"aec", "", "Continue until exception",
	"aecs", "", "Continue until syscall",
	"aecu", "[addr]", "Continue until address",
	"aecue", "[addr]", "Continue until esil expression",
	NULL
};

static const char *help_msg_aep[] = {
	"Usage:", "aep[-c] ", " [...]",
	"aepc", " [addr]", "change program counter for esil",
	"aep", "-[addr]", "remove pin",
	"aep", " [name] @ [addr]", "set pin",
	"aep", "", "list pins",
	NULL
};

static const char *help_msg_aets[] = {
	"Usage:", "aets ", " [...]",
	"aets", "", "List all ESIL trace sessions",
	"aets+", "", "Add ESIL trace session",
	NULL
};

static const char *help_msg_af[] = {
	"Usage:", "af", "",
	"af", " ([name]) ([addr])", "analyze functions (start at addr or $$)",
	"afr", " ([name]) ([addr])", "analyze functions recursively",
	"af+", " addr name [type] [diff]", "hand craft a function (requires afb+)",
	"af-", " [addr]", "clean all function analysis data (or function at addr)",
	"afb+", " fcnA bbA sz [j] [f] ([t]( [d]))", "add bb to function @ fcnaddr",
	"afb", "[?] [addr]", "List basic blocks of given function",
	"afB", " 16", "set current function as thumb (change asm.bits)",
	"afC[lc]", " ([addr])@[addr]", "calculate the Cycles (afC) or Cyclomatic Complexity (afCc)",
	"afc", "[?] type @[addr]", "set calling convention for function",
	"afd", "[addr]","show function + delta for given offset",
	"aft", "[?]", "type matching, type propagation",
	"aff", "", "re-adjust function boundaries to fit",
	"afF", "[1|0|]", "fold/unfold/toggle",
	"afi", " [addr|fcn.name]", "show function(s) information (verbose afl)",
	"afl", "[?] [l*] [fcn name]", "list functions (addr, size, bbs, name) (see afll)",
	"afo", " [fcn.name]", "show address for the function named like this",
	"afm", " name", "merge two functions",
	"afM", " name", "print functions map",
	"afn", "[?] name [addr]", "rename name for function at address (change flag too)",
	"afna", "", "suggest automatic name for current offset",
	"afs", " [addr] [fcnsign]", "get/set function signature at current address",
	"afS", "[stack_size]", "set stack frame size for function at current address",
	"afu", " [addr]", "resize and analyze function from current address until addr",
	"afv[bsra]", "?", "manipulate args, registers and variables in function",
	"afx", "[cCd-] src dst", "add/remove code/Call/data/string reference",
	NULL
};

static const char *help_msg_afb[] = {
	"Usage:", "afb", " List basic blocks of given function",
	".afbr-", "", "Set breakpoint on every return address of the function",
	".afbr-*", "", "Remove breakpoint on every return address of the function",
	"afb", " [addr]", "list basic blocks of function",
	"afb.", " [addr]", "show info of current basic block",
	"afb+", " fcn_at bbat bbsz [jump] [fail] ([type] ([diff]))", "add basic block by hand",
	"afbr", "", "Show addresses of instructions which leave the function",
	"afbi", "", "print current basic block information",
	"afbj", "", "show basic blocks information in json",
	"afbe", " bbfrom bbto", "add basic-block edge for switch-cases",
	"afB", " [bits]", "define asm.bits for the given function",
	NULL
};

static const char *help_msg_afc[] = {
	"Usage:", "afc[agl?]", "",
	"afc", " convention", "Manually set calling convention for current function",
	"afc", "", "Show Calling convention for the Current function",
	"afcr", "[j]", "Show register usage for the current function",
	"afca", "", "Analyse function for finding the current calling convention",
	"afcl", "", "List all available calling conventions",
	"afco", " path", "Open Calling Convention sdb profile from given path",
	NULL
};

static const char *help_msg_afC[] = {
	"Usage:", "afC", " [addr]",
	"afC", "", "function cycles cost",
	"afCc", "", "cyclomatic complexity",
	"afCl", "", "loop count (backward jumps)",
	NULL
};

static const char *help_msg_afi[] = {
	"Usage:", "afi[jl*]", " <addr>",
	"afi", "", "show information of the function",
	"afi.", "", "show function name in current offset",
	"afi*", "", "function, variables and arguments",
	"afij", "", "function info in json format",
	"afil", "", "verbose function info",
	NULL
};

static const char *help_msg_afl[] = {
	"Usage:", "afl", " List all functions",
	"afl", "", "list functions",
	"aflc", "", "count of functions",
	"aflj", "", "list functions in json",
	"afll", "", "list functions in verbose mode",
	"afllj", "", "list functions in verbose mode (alias to aflj)",
	"aflq", "", "list functions in quiet mode",
	"aflqj", "", "list functions in json quiet mode",
	"afls", "", "print sum of sizes of all functions",
	NULL
};

static const char *help_msg_afll[] = {
	"Usage:", "", " List functions in verbose mode",
	"", "", "",
	"Table fields:", "", "",
	"", "", "",
	"address", "", "start address",
	"size", "", "function size (realsize)",
	"nbbs", "", "number of basic blocks",
	"edges", "", "number of edges between basic blocks",
	"cc", "", "cyclomatic complexity ( cc = edges - blocks + 2 * exit_blocks)",
	"cost", "", "cyclomatic cost",
	"min bound", "", "minimal address",
	"range", "", "function size",
	"max bound", "", "maximal address",
	"calls", "", "number of caller functions",
	"locals", "", "number of local variables",
	"args", "", "number of function arguments",
	"xref", "", "number of cross references",
	"frame", "", "function stack size",
	"name", "", "function name",
	NULL
};

static const char *help_msg_afn[] = {
	"Usage:", "afn[sa]", " Analyze function names",
	"afn", " [name]", "rename the function",
	"afna", "", "construct a function name for the current offset",
	"afns", "", "list all strings associated with the current function",
	NULL
};

static const char *help_msg_aft[] = {
	"Usage:", "aftm", "",
	"afta", "", "Setup memory and analyse do type matching analysis for all functions",
	"aftm", "", "type matching analysis",
	NULL
};

static const char *help_msg_afv[] = {
	"Usage:", "afv","[rbs]",
	"afvr", "[?]", "manipulate register based arguments",
	"afvb", "[?]", "manipulate bp based arguments/locals",
	"afvs", "[?]", "manipulate sp based arguments/locals",
	"afvR", " [varname]", "list addresses where vars are accessed",
	"afvW", " [varname]", "list addresses where vars are accessed",
	"afva", "", "analyze function arguments/locals",
	"afvd", " name", "output r2 command for displaying the value of args/locals in the debugger",
	"afvn", " [old_name] [new_name]", "rename argument/local",
	"afvt", " [name] [new_type]", "change type for given argument/local",
	"afv-", "([name])", "remove all or given var",
	NULL
};

static const char *help_msg_afvb[] = {
	"Usage:", "afvb", " [idx] [name] ([type])",
	"afvb", "", "list base pointer based arguments, locals",
	"afvb*", "", "same as afvb but in r2 commands",
	"afvb", " [idx] [name] ([type])", "define base pointer based arguments, locals",
	"afvbj", "", "return list of base pointer based arguments, locals in JSON format",
	"afvb-", " [name]", "delete argument/locals at the given name",
	"afvbg", " [idx] [addr]", "define var get reference",
	"afvbs", " [idx] [addr]", "define var set reference",
	NULL
};

static const char *help_msg_afvr[] = {
	"Usage:", "afvr", " [reg] [type] [name]",
	"afvr", "", "list register based arguments",
	"afvr*", "", "same as afvr but in r2 commands",
	"afvr", " [reg] [name] ([type])", "define register arguments",
	"afvrj", "", "return list of register arguments in JSON format",
	"afvr-", " [name]", "delete register arguments at the given index",
	"afvrg", " [reg] [addr]", "define argument get reference",
	"afvrs", " [reg] [addr]", "define argument set reference",
	NULL
};

static const char *help_msg_afvs[] = {
	"Usage:", "afvs", " [idx] [type] [name]",
	"afvs", "", "list stack based arguments and locals",
	"afvs*", "", "same as afvs but in r2 commands",
	"afvs", " [idx] [name] [type]", "define stack based arguments,locals",
	"afvsj", "", "return list of stack based arguments and locals in JSON format",
	"afvs-", " [name]", "delete stack based argument or locals with the given name",
	"afvsg", " [idx] [addr]", "define var get reference",
	"afvss", " [idx] [addr]", "define var set reference",
	NULL
};

static const char *help_msg_afx[] = {
	"Usage:", "afx[-cCd?] [src] [dst]", " manage function references (see also ar?)",
	"afxc", " sym.main+0x38 sym.printf", "add code ref",
	"afxC", " sym.main sym.puts", "add call ref",
	"afxd", " sym.main str.helloworld", "add data ref",
	"afx-", " sym.main str.helloworld", "remove reference",
	NULL
};

static const char *help_msg_ag[] = {
	"Usage:", "ag[?f]", " Graphviz/graph code",
	"ag", " [addr]", "output graphviz code (bb at addr and children)",
	"ag-", "", "Reset the current ASCII art graph (see agn, age, agg?)",
	"aga", " [addr]", "idem, but only addresses",
	"agr", "[j] [addr]", "output graphviz call graph of function",
	"agg", "", "display current graph created with agn and age (see also ag-)",
	"agc", "[*j] [addr]", "output graphviz call graph of function",
	"agC", "[j]", "Same as agc -1. full program callgraph",
	"agd", " [fcn name]", "output graphviz code of diffed function",
	"age", "[?] title1 title2", "Add an edge to the current graph",
	"agf", " [addr]", "Show ASCII art graph of given function",
	"agg", "[?] [kdi*]", "Print graph in ASCII-Art, graphviz, k=v, r2 or visual",
	"agj", " [addr]", "idem, but in JSON format",
	"agJ", " [addr]", "idem, but in JSON format with formatted disassembly (like pdJ)",
	"agk", " [addr]", "idem, but in SDB key-value format",
	"agl", " [fcn name]", "output graphviz code using meta-data",
	"agn", "[?] title body", "Add a node to the current graph",
	"ags", " [addr]", "output simple graphviz call graph of function (only bb offset)",
	"agt", " [addr]", "find paths from current offset to given address",
	"agv", "", "Show function graph in web/png (see graph.web and cmd.graph) or agf for asciiart",
	NULL
};

static const char *help_msg_age[] = {
	"Usage:", "age [title1] [title2]", "",
	"Examples:", "", "",
	"age", " title1 title2", "Add an edge from the node with \"title1\" as title to the one with title \"title2\"",
	"age", " \"title1 with spaces\" title2", "Add an edge from node \"title1 with spaces\" to node \"title2\"",
	"age-", " title1 title2", "Remove an edge from the node with \"title1\" as title to the one with title \"title2\"",
	"age?", "", "Show this help",
	NULL
};

static const char *help_msg_agg[] = {
	"Usage:", "agg[kid?*]", "print graph",
	"agg", "", "show current graph in ascii art",
	"aggk", "", "show graph in key=value form",
	"aggi", "", "enter interactive mode for the current graph",
	"aggd", "", "print the current graph in GRAPHVIZ dot format",
	"aggv", "", "run graphviz + viewer (see 'e cmd.graph')",
	"agg*", "", "in r2 commands, to save in projects, etc",
	NULL
};

static const char *help_msg_agn[] = {
	"Usage:", "agn [title] [body]", "",
	"Examples:", "", "",
	"agn", " title1 body1", "Add a node with title \"title1\" and body \"body1\"",
	"agn", " \"title with space\" \"body with space\"", "Add a node with spaces in the title and in the body",
	"agn", " title1 base64:Ym9keTE=", "Add a node with the body specified as base64",
	"agn-", " title1", "Remove a node with title \"title1\"",
	"agn?", "", "Show this help",
	NULL
};

static const char *help_msg_ah[] = {
	"Usage:", "ah[lba-]", "Analysis Hints",
	"ah?", "", "show this help",
	"ah?", " offset", "show hint of given offset",
	"ah", "", "list hints in human-readable format",
	"ah.", "", "list hints in human-readable format from current offset",
	"ah-", "", "remove all hints",
	"ah-", " offset [size]", "remove hints at given offset",
	"ah*", " offset", "list hints in radare commands format",
	"aha", " ppc 51", "set arch for a range of N bytes",
	"ahb", " 16 @ $$", "force 16bit for current instruction",
	"ahc", " 0x804804", "override call/jump address",
	"ahe", " 3,eax,+=", "set vm analysis string",
	"ahf", " 0x804840", "override fallback address for call",
	"ahh", " 0x804840", "highlight this adrress offset in disasm",
	"ahi", "[?] 10", "define numeric base for immediates (1, 8, 10, 16, s)",
	"ahj", "", "list hints in JSON",
	"aho", " foo a0,33", "replace opcode string",
	"ahp", " addr", "set pointer hint",
	"ahs", " 4", "set opcode size=4",
	"ahS", " jz", "set asm.syntax=jz for this opcode",
	NULL
};

static const char *help_msg_ahi[] = {
	"Usage", "ahi [sbodh] [@ offset]", " Define numeric base",
	"ahi", " [base]", "set numeric base (1, 2, 8, 10, 16)",
	"ahi", " b", "set base to binary (2)",
	"ahi", " d", "set base to decimal (10)",
	"ahi", " h", "set base to hexadecimal (16)",
	"ahi", " o", "set base to octal (8)",
	"ahi", " p", "set base to htons(port) (3)",
	"ahi", " i", "set base to IP address (32)",
	"ahi", " S", "set base to syscall (80)",
	"ahi", " s", "set base to string (1)",
	NULL
};

static const char *help_msg_ao[] = {
	"Usage:", "ao[e?] [len]", "Analyze Opcodes",
	"aoj", " N", "display opcode analysis information in JSON for N opcodes",
	"aoe", " N", "display esil form for N opcodes",
	"aor", " N", "display reil form for N opcodes",
	"aos", " N", "display size of N opcodes",
	"ao", " 5", "display opcode analysis of 5 opcodes",
	"ao*", "", "display opcode in r commands",
	NULL
};

static const char *help_msg_ar[] = {
	"Usage: ar", "", "# Analysis Registers",
	"ar", "", "Show 'gpr' registers",
	"ar0", "", "Reset register arenas to 0",
	"ara", "[?]", "Manage register arenas",
	"ar", " 16", "Show 16 bit registers",
	"ar", " 32", "Show 32 bit registers",
	"ar", " all", "Show all bit registers",
	"ar", " <type>", "Show all registers of given type",
	"arC", "", "Display register profile comments",
	"arr", "", "Show register references (telescoping)",
	"ar=", "([size])(:[regs])", "Show register values in columns",
	"ar?", " <reg>", "Show register value",
	"arb", " <type>", "Display hexdump of the given arena",
	"arc", " <name>", "Conditional flag registers",
	"ard", " <name>", "Show only different registers",
	"arn", " <regalias>", "Get regname for pc,sp,bp,a0-3,zf,cf,of,sg",
	"aro", "", "Show old (previous) register values",
	"arp", "[?] <file>", "Load register profile from file",
	"ars", "", "Stack register state",
	"art", "", "List all register types",
	"arw", " <hexnum>", "Set contents of the register arena",
	".ar*", "", "Import register values as flags",
	".ar-", "", "Unflag all registers",
	NULL
};

static const char *help_msg_ara[] = {
	"Usage:", "ara[+-s]", "Register Arena Push/Pop/Swap",
	"ara", "", "show all register arenas allocated",
	"ara", "+", "push a new register arena for each type",
	"ara", "-", "pop last register arena",
	"aras", "", "swap last two register arenas",
	NULL
};

static const char *help_msg_arw[] = {
	"Usage:", "arw ", "# Set contents of the register arena",
	"arw", " <hexnum>", "Set contents of the register arena",
	NULL
};

static const char *help_msg_as[] = {
	"Usage: as[ljk?]", "", "syscall name <-> number utility",
	"as", "", "show current syscall and arguments",
	"as", " 4", "show syscall 4 based on asm.os and current regs/mem",
	"asc[a]", " 4", "dump syscall info in .asm or .h",
	"asf", " [k[=[v]]]", "list/set/unset pf function signatures (see fcnsign)",
	"asj", "", "list of syscalls in JSON",
	"asl", "", "list of syscalls by asm.os and asm.arch",
	"asl", " close", "returns the syscall number for close",
	"asl", " 4", "returns the name of the syscall number 4",
	"ask", " [query]", "perform syscall/ queries",
	NULL
};

static const char *help_msg_av[] = {
	"Usage:", "av[?jr*]", " C++ vtables and RTTI",
	"av", "", "search for vtables in data sections and show results",
	"avj", "", "like av, but as json",
	"av*", "", "like av, but as r2 commands",
	"avr", "[j@addr]", "try to parse RTTI at vtable addr (see anal.cpp.abi)",
	"avra", "[j]", "search for vtables and try to parse RTTI at each of them",
	NULL
};

static const char *help_msg_ax[] = {
	"Usage:", "ax[?d-l*]", " # see also 'afx?'",
	"ax", "", "list refs",
	"ax", " addr [at]", "add code ref pointing to addr (from curseek)",
	"ax-", " [at]", "clean all refs (or refs from addr)",
	"axc", " addr [at]", "add code jmp ref // unused?",
	"axC", " addr [at]", "add code call ref",
	"axg", " [addr]", "show xrefs graph to reach current function",
	"axgj", " [addr]", "show xrefs graph to reach current function in json format",
	"axd", " addr [at]", "add data ref",
	"axq", "", "list refs in quiet/human-readable format",
	"axj", "", "list refs in json format",
	"axF", " [flg-glob]", "find data/code references of flags",
	"axt", " [addr]", "find data/code references to this address",
	"axf", " [addr]", "find data/code references from this address",
	"axk", " [query]", "perform sdb query",
	"ax*", "", "output radare commands",
	NULL
};

static void cmd_anal_init(RCore *core) {
	DEFINE_CMD_DESCRIPTOR (core, a);
	DEFINE_CMD_DESCRIPTOR (core, aa);
	DEFINE_CMD_DESCRIPTOR (core, aar);
	DEFINE_CMD_DESCRIPTOR (core, ab);
	DEFINE_CMD_DESCRIPTOR (core, ad);
	DEFINE_CMD_DESCRIPTOR (core, ae);
	DEFINE_CMD_DESCRIPTOR (core, aea);
	DEFINE_CMD_DESCRIPTOR (core, aec);
	DEFINE_CMD_DESCRIPTOR (core, aep);
	DEFINE_CMD_DESCRIPTOR (core, af);
	DEFINE_CMD_DESCRIPTOR (core, afb);
	DEFINE_CMD_DESCRIPTOR (core, afc);
	DEFINE_CMD_DESCRIPTOR (core, afC);
	DEFINE_CMD_DESCRIPTOR (core, afi);
	DEFINE_CMD_DESCRIPTOR (core, afl);
	DEFINE_CMD_DESCRIPTOR (core, afll);
	DEFINE_CMD_DESCRIPTOR (core, afn);
	DEFINE_CMD_DESCRIPTOR (core, aft);
	DEFINE_CMD_DESCRIPTOR (core, afv);
	DEFINE_CMD_DESCRIPTOR (core, afvb);
	DEFINE_CMD_DESCRIPTOR (core, afvr);
	DEFINE_CMD_DESCRIPTOR (core, afvs);
	DEFINE_CMD_DESCRIPTOR (core, afx);
	DEFINE_CMD_DESCRIPTOR (core, ag);
	DEFINE_CMD_DESCRIPTOR (core, age);
	DEFINE_CMD_DESCRIPTOR (core, agg);
	DEFINE_CMD_DESCRIPTOR (core, agn);
	DEFINE_CMD_DESCRIPTOR (core, ah);
	DEFINE_CMD_DESCRIPTOR (core, ahi);
	DEFINE_CMD_DESCRIPTOR (core, ao);
	DEFINE_CMD_DESCRIPTOR (core, ar);
	DEFINE_CMD_DESCRIPTOR (core, ara);
	DEFINE_CMD_DESCRIPTOR (core, arw);
	DEFINE_CMD_DESCRIPTOR (core, as);
	DEFINE_CMD_DESCRIPTOR (core, ax);
}

/* better aac for windows-x86-32 */
#define JAYRO_03 0

#if JAYRO_03

static bool anal_is_bad_call(RCore *core, ut64 from, ut64 to, ut64 addr, ut8 *buf, int bufi) {
	ut64 align = R_ABS (addr % PE_ALIGN);
	ut32 call_bytes;

	// XXX this is x86 specific
	if (align == 0) {
		call_bytes = (ut32)((ut8*)buf)[bufi + 3] << 24;
		call_bytes |= (ut32)((ut8*)buf)[bufi + 2] << 16;
		call_bytes |= (ut32)((ut8*)buf)[bufi + 1] << 8;
		call_bytes |= (ut32)((ut8*)buf)[bufi];
	} else {
		call_bytes = (ut32)((ut8*)buf)[bufi - align + 3] << 24;
		call_bytes |= (ut32)((ut8*)buf)[bufi - align + 2] << 16;
		call_bytes |= (ut32)((ut8*)buf)[bufi - align + 1] << 8;
		call_bytes |= (ut32)((ut8*)buf)[bufi - align];
	}
	if (call_bytes >= from && call_bytes <= to) {
		return true;
	}
	call_bytes = (ut32)((ut8*)buf)[bufi + 4] << 24;
	call_bytes |= (ut32)((ut8*)buf)[bufi + 3] << 16;
	call_bytes |= (ut32)((ut8*)buf)[bufi + 2] << 8;
	call_bytes |= (ut32)((ut8*)buf)[bufi + 1];
	call_bytes += addr + 5;
	if (call_bytes >= from && call_bytes <= to) {
		return false;
	}
	return false;
}
#endif

static void type_cmd(RCore *core, const char *input) {
	RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
	if (!fcn && *input != '?' && *input != 'a') {
		eprintf ("cant find function here\n");
		return;
	}
	RListIter *it;
	ut64 seek;
	bool io_cache =  r_config_get_i (core->config, "io.cache");
	r_cons_break_push (NULL, NULL);
	switch (*input) {
	case 'a': // "afta"
		if (r_config_get_i (core->config, "cfg.debug")) {
			eprintf ("TOFIX: afta can't run in debugger mode.\n");
			break;
		}
		seek = core->offset;
		r_core_cmd0 (core, "aei");
		r_core_cmd0 (core, "aeim");
		r_reg_arena_push (core->anal->reg);
		r_list_foreach (core->anal->fcns, it, fcn) {
			int ret = r_core_seek (core, fcn->addr, true);
			if (!ret) {
				continue;
			}
			r_anal_esil_set_pc (core->anal->esil, fcn->addr);
			r_core_anal_type_match (core, fcn);
			if (r_cons_is_breaked ()) {
				break;
			}
		}
		r_core_cmd0 (core, "aeim-");
		r_core_cmd0 (core, "aei-");
		r_core_seek (core, seek, true);
		r_reg_arena_pop (core->anal->reg);
		break;
	case 'm': // "aftm"
		r_config_set_i (core->config, "io.cache", true);
		seek = core->offset;
		r_anal_esil_set_pc (core->anal->esil, fcn? fcn->addr: core->offset);
		r_core_anal_type_match (core, fcn);
		r_core_seek (core, seek, true);
		r_config_set_i (core->config, "io.cache", io_cache);
		break;
	case '?':
		r_core_cmd_help (core, help_msg_aft);
		break;
	}
	r_cons_break_pop ();
}

static int cc_print(void *p, const char *k, const char *v) {
	if (!strcmp (v, "cc")) {
		r_cons_println (k);
	}
	return 1;
}

static void find_refs(RCore *core, const char *glob) {
	char cmd[128];
	ut64 curseek = core->offset;
	while (*glob == ' ') glob++;
	if (!*glob) {
		glob = "str.";
	}
	if (*glob == '?') {
		eprintf ("Usage: arf [flag-str-filter]\n");
		return;
	}
	eprintf ("Finding references of flags matching '%s'...\n", glob);
	snprintf (cmd, sizeof (cmd) - 1, ".(findstref) @@= `f~%s[0]`", glob);
	r_core_cmd0 (core, "(findstref,f here=$$,s entry0,/r here,f-here)");
	r_core_cmd0 (core, cmd);
	r_core_cmd0 (core, "(-findstref)");
	r_core_seek (core, curseek, 1);
}

/* set flags for every function */
static void flag_every_function(RCore *core) {
	RListIter *iter;
	RAnalFunction *fcn;
	r_flag_space_push (core->flags, "functions");
	r_list_foreach (core->anal->fcns, iter, fcn) {
		r_flag_set (core->flags, fcn->name,
			fcn->addr, r_anal_fcn_size (fcn));
	}
	r_flag_space_pop (core->flags);
}

static void var_help(RCore *core, char ch) {
	switch (ch) {
	case 'b':
		r_core_cmd_help (core, help_msg_afvb);
		break;
	case 's':
		r_core_cmd_help (core, help_msg_afvs);
		break;
	case 'r':
		r_core_cmd_help (core, help_msg_afvr);
		break;
	case '?':
		r_core_cmd_help (core, help_msg_afv);
		break;
	default:
		eprintf ("See afv?, afvb?, afvr? and afvs?\n");
	}
}

static void var_accesses_list(RAnal *a, RAnalFunction *fcn, int delta, const char *typestr) {
	const char *var_local = sdb_fmt ("var.0x%"PFMT64x".%d.%d.%s",
			fcn->addr, 1, delta, typestr);
	const char *xss = sdb_const_get (a->sdb_fcns, var_local, 0);
	if (xss && *xss) {
		r_cons_printf ("%s\n", xss);
	} else {
		r_cons_newline ();
	}
}

static void list_vars(RCore *core, RAnalFunction *fcn, int type, const char *name) {
	RAnalVar *var;
	RListIter *iter;
	RList *list = r_anal_var_list (core->anal, fcn, 0);
	if (type == '*') {
		const char *bp = r_reg_get_name (core->anal->reg, R_REG_NAME_BP);
		r_cons_printf ("f-fcnvar*\n");
		r_list_foreach (list, iter, var) {
			r_cons_printf ("f fcnvar.%s @ %s%s%d\n", var->name, bp,
				var->delta>=0? "+":"", var->delta);
		}
		return;
	}
	if (type != 'W' && type != 'R') {
		return;
	}
	const char *typestr = type == 'R'?"reads":"writes";
	r_list_foreach (list, iter, var) {
		r_cons_printf ("%10s  ", var->name);
		var_accesses_list (core->anal, fcn, var->delta, typestr);
	}
}

static int cmd_an(RCore *core, bool use_json, const char *name)
{
	ut64 off = core->offset;
	RAnalOp op;
	char *q = NULL;
	ut64 tgt_addr = UT64_MAX;

	if (use_json) {
		r_cons_print ("[");
	}

	r_anal_op (core->anal, &op, off,
			core->block + off - core->offset, 32, R_ANAL_OP_MASK_ALL);

	tgt_addr = op.jump != UT64_MAX ? op.jump : op.ptr;
	if (op.var) {
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, off, 0);
		if (fcn) {
			RAnalVar *bar = r_anal_var_get_byname (core->anal, fcn, op.var->name);
			if (!bar) {
				bar = r_anal_var_get_byname (core->anal, fcn, op.var->name);
				if (!bar) {
					bar = r_anal_var_get_byname (core->anal, fcn, op.var->name);
				}
			}
			if (bar) {
				if (name) {
					r_anal_var_rename (core->anal, fcn->addr, bar->scope,
									bar->kind, bar->name, name);
				} else if (!use_json) {
					r_cons_println (bar->name);
				} else {
					r_cons_printf ("{\"type\":\"var\",\"name\":\"%s\"}",
								bar->name);
				}
			} else {
				eprintf ("Cannot find variable\n");
			}
		} else {
			eprintf ("Cannot find function\n");
		}
	} else if (tgt_addr != UT64_MAX) {
		RAnalFunction *fcn = r_anal_get_fcn_at (core->anal, tgt_addr, R_ANAL_FCN_TYPE_NULL);
		RFlagItem *f = r_flag_get_i (core->flags, tgt_addr);
		if (fcn) {
			if (name) {
				q = r_str_newf ("afn %s 0x%"PFMT64x, name, tgt_addr);
			} else if (!use_json) {
				r_cons_println (fcn->name);
			} else {
				r_cons_printf ("{\"type\":\"function\",\"name\":\"%s\"}",
							fcn->name);
			}
		} else if (f) {
			if (name) {
				q = r_str_newf ("fr %s %s", f->name, name);
			} else if (!use_json) {
				r_cons_println (f->name);
			} else {
				r_cons_printf ("{\"type\":\"flag\",\"name\":\"%s\"}",
							f->name);
			}
		} else {
			if (name) {
				q = r_str_newf ("f %s @ 0x%"PFMT64x, name, tgt_addr);
			} else if (!use_json) {
				r_cons_printf ("0x%" PFMT64x "\n", tgt_addr);
			} else {
				r_cons_printf ("{\"type\":\"address\",\"offset\":"
							   "%" PFMT64d "}", tgt_addr);
			}
		}
	}

	if (use_json) {
		r_cons_print ("]\n");
	}

	if (q) {
		r_core_cmd0 (core, q);
		free (q);
	}
	r_anal_op_fini (&op);
	return 0;
}

static int var_cmd(RCore *core, const char *str) {
	char *p, *ostr;
	int delta, type = *str, res = true;
	RAnalVar *v1;
	RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
	ostr = p = NULL;
	if (!str[0]) {
		// "afv"
		if (fcn) {
			r_core_cmd0 (core, "afvs");
			r_core_cmd0 (core, "afvb");
			r_core_cmd0 (core, "afvr");
			return true;
		}
		eprintf ("Cannot find function\n");
		return false;
	}
	if (str[0] == 'j') {
		// "afvj"
		if (fcn) {
			r_cons_printf ("{\"sp\":");
			r_core_cmd0 (core, "afvsj");
			r_cons_printf (",\"bp\":");
			r_core_cmd0 (core, "afvbj");
			r_cons_printf (",\"reg\":");
			r_core_cmd0 (core, "afvrj");
			r_cons_printf ("}\n");
			return true;
		}
		eprintf ("Cannot find function\n");
		return false;
	}
	if (!str[0] || str[1] == '?'|| str[0] == '?') {
		var_help (core, *str);
		return res;
	}
	if (!fcn) {
		eprintf ("Cannot find function in 0x%08"PFMT64x"\n", core->offset);
		return false;
	}
	ostr = p = strdup (str);
	/* Variable access CFvs = set fun var */
	switch (str[0]) {
	case '-':
		// "afv"
		if (fcn) {
			r_core_cmdf (core, "afvs-%s", str + 1);
			r_core_cmdf (core, "afvb-%s", str + 1);
			r_core_cmdf (core, "afvr-%s", str + 1);
			return true;
		}
		eprintf ("Cannot find function\n");
		return false;
	case 'R': // "afvR"
	case 'W': // "afvW"
	case '*': // "afv*"
		list_vars (core, fcn, str[0], str + 1);
		break;
	case 'a': // "afva"
		r_anal_var_delete_all (core->anal, fcn->addr, R_ANAL_VAR_KIND_REG);
		r_anal_var_delete_all (core->anal, fcn->addr, R_ANAL_VAR_KIND_BPV);
		r_anal_var_delete_all (core->anal, fcn->addr, R_ANAL_VAR_KIND_SPV);
		fcn_callconv (core, fcn);
		free (p);
		return true;
	case 'n':
		if (str[1]) { // "afvn"
			char *old_name = r_str_trim_head (strchr (ostr, ' '));
			if (!old_name) {
				free (ostr);
				return false;
			}
			char *new_name = strchr (old_name, ' ');
			if (!new_name) {
				free (ostr);
				return false;
			}
			*new_name++ = 0;
			r_str_trim (new_name);
			v1 = r_anal_var_get_byname (core->anal, fcn, old_name);
			if (v1) {
				r_anal_var_rename (core->anal, fcn->addr, R_ANAL_VAR_SCOPE_LOCAL,
						v1->kind, old_name, new_name);
				r_anal_var_free (v1);
			} else {
				eprintf ("Cant find var by name\n");
			}
			free (ostr);
		} else {
			RListIter *iter;
			RAnalVar *v;
			RList *list = r_anal_var_list (core->anal, fcn, 0);
			r_list_foreach (list, iter, v) {
				r_cons_printf ("%s\n", v->name);
			}
		}
		return true;
	case 'd': // "afvd"
		if (str[1]) {
			p = r_str_trim (strchr (ostr, ' '));
			if (!p) {
				free (ostr);
				return false;
			}
			v1 = r_anal_var_get_byname (core->anal, fcn, p);
			if (!v1) {
				free (ostr);
				return false;
			}
			r_anal_var_display (core->anal, v1->delta, v1->kind, v1->type);
			r_anal_var_free (v1);
			free (ostr);
		} else {
			RListIter *iter;
			RAnalVar *p;
			RList *list = r_anal_var_list (core->anal, fcn, 0);
			r_list_foreach (list, iter, p) {
				char *a = r_core_cmd_strf (core, ".afvd %s", p->name);
				if ((a && !*a) || !a) {
					free (a);
					a = strdup ("\n");
				}
				r_cons_printf ("var %s = %s", p->name, a);
				free (a);
			}
			r_list_free (list);
			// args
			list = r_anal_var_list (core->anal, fcn, 1);
			r_list_foreach (list, iter, p) {
				char *a = r_core_cmd_strf (core, ".afvd %s", p->name);
				if ((a && !*a) || !a) {
					free (a);
					a = strdup ("\n");
				}
				r_cons_printf ("arg %s = %s", p->name, a);
				free (a);
			}
			r_list_free (list);
		}
		return true;
	case 't':{ // "afvt"
		p = strchr (ostr, ' ');
		if (!p++) {
			free (ostr);
			return false;
		}

		char *type = strchr (p, ' ');
		if (!type) {
			free (ostr);
			return false;
		}
		*type++ = 0;
		v1 = r_anal_var_get_byname (core->anal, fcn, p);
		if (!v1) {
			eprintf ("Cant find get by name %s\n", p);
			free (ostr);
			return false;
		}
		r_anal_var_retype (core->anal, fcn->addr,
			R_ANAL_VAR_SCOPE_LOCAL, -1, v1->kind, type, -1, p);
		r_anal_var_free (v1);
		free (ostr);
		return true;

	}
	}
	switch (str[1]) {
	case '\0':
	case '*':
	case 'j':
		r_anal_var_list_show (core->anal, fcn, type, str[1]);
		break;
	case '.':
		r_anal_var_list_show (core->anal, fcn, core->offset, 0);
		break;
	case '-': // "afv[bsr]-"
		if (str[2] == '*') {
			r_anal_var_delete_all (core->anal, fcn->addr, type);
		} else {
			if (IS_DIGIT (str[2])) {
				r_anal_var_delete (core->anal, fcn->addr,
						type, 1, (int)r_num_math (core->num, str + 1));
			} else {
				char *name = r_str_trim ( strdup (str + 2));
				r_anal_var_delete_byname (core->anal, fcn, type, name);
				free (name);
			}
		}
		break;
	case 'd':
		eprintf ("This command is deprecated, use afvd instead\n");
		break;
	case 't':
		eprintf ("This command is deprecated use afvt instead\n");
		break;
	case 's':
	case 'g':
		if (str[2] != '\0') {
			int rw = 0; // 0 = read, 1 = write
			RAnalVar *var = r_anal_var_get (core->anal, fcn->addr,
							(char)type, atoi (str + 2), R_ANAL_VAR_SCOPE_LOCAL);
			if (!var) {
				eprintf ("Cannot find variable in: '%s'\n", str);
				res = false;
				break;
			}
			if (var != NULL) {
				int scope = (str[1] == 'g')? 0: 1;
				r_anal_var_access (core->anal, fcn->addr, (char)type,
						scope, atoi (str + 2), rw, core->offset);
				r_anal_var_free (var);
				break;
			}
		} else {
			eprintf ("Missing argument\n");
		}
		break;
	case ' ': {
		const char *name;
		char *vartype;
		int size = 4;
		int scope = 1;
			for (str++; *str == ' ';) str++;
		p = strchr (str, ' ');
		if (!p) {
			var_help (core, type);
			break;
		}
		*p++ = 0;
		if (type == 'r') { //registers
			RRegItem *i = r_reg_get (core->anal->reg, str, -1);
			if (!i) {
				eprintf ("Register not found");
				break;
			}
			delta = i->index;
		} else {
			delta = r_num_math (core->num, str);
		}
		name = p;
		vartype = strchr (name, ' ');
		if (vartype) {
			*vartype++ = 0;
			r_anal_var_add (core->anal, fcn->addr,
					scope, delta, type,
					vartype, size, name);
		} else {
			eprintf ("Missing name\n");
		}
		}
		break;
	};

	free (ostr);
	return res;
}

static void print_trampolines(RCore *core, ut64 a, ut64 b, size_t element_size) {
	int i;
	for (i = 0; i < core->blocksize; i += element_size) {
		ut32 n;
		memcpy (&n, core->block + i, sizeof (ut32));
		if (n >= a && n <= b) {
			if (element_size == 4) {
				r_cons_printf ("f trampoline.%x @ 0x%" PFMT64x "\n", n, core->offset + i);
			} else {
				r_cons_printf ("f trampoline.%" PFMT64x " @ 0x%" PFMT64x "\n", n, core->offset + i);
			}
			r_cons_printf ("Cd %u @ 0x%" PFMT64x ":%u\n", element_size, core->offset + i, element_size);
			// TODO: add data xrefs
		}
	}
}

static void cmd_anal_trampoline(RCore *core, const char *input) {
	int bits = r_config_get_i (core->config, "asm.bits");
	char *p, *inp = strdup (input);
	p = strchr (inp, ' ');
	if (p) {
		*p = 0;
	}
	ut64 a = r_num_math (core->num, inp);
	ut64 b = p? r_num_math (core->num, p + 1): 0;
	free (inp);

	switch (bits) {
	case 32:
		print_trampolines (core, a, b, 4);
		break;
	case 64:
		print_trampolines (core, a, b, 8);
		break;
	}
}

R_API char *cmd_syscall_dostr(RCore *core, int n) {
	char *res = NULL;
	int i;
	char str[64];
	if (n == -1) {
		n = (int)r_debug_reg_get (core->dbg, "oeax");
		if (!n || n == -1) {
			const char *a0 = r_reg_get_name (core->anal->reg, R_REG_NAME_SN);
			n = (int)r_debug_reg_get (core->dbg, a0);
		}
	}
	RSyscallItem *item = r_syscall_get (core->anal->syscall, n, -1);
	if (!item) {
		res = r_str_appendf (res, "%d = unknown ()", n);
		return res;
	}
	res = r_str_appendf (res, "%d = %s (", item->num, item->name);
	// TODO: move this to r_syscall
	//TODO replace the hardcoded CC with the sdb ones
	for (i = 0; i < item->args; i++) {
		// XXX this is a hack to make syscall args work on x86-32 and x86-64
		// we need to shift sn first.. which is bad, but needs to be redesigned
		int regidx = i;
		if (core->assembler->bits == 32) {
			regidx++;
		}
		ut64 arg = r_debug_arg_get (core->dbg, R_ANAL_CC_TYPE_FASTCALL, regidx);
		//r_cons_printf ("(%d:0x%"PFMT64x")\n", i, arg);
		if (item->sargs) {
			switch (item->sargs[i]) {
			case 'p': // pointer
				res = r_str_appendf (res, "0x%08" PFMT64x "", arg);
				break;
			case 'i':
				res = r_str_appendf (res, "%" PFMT64d "", arg);
				break;
			case 'z':
				memset (str, 0, sizeof (str));
				r_io_read_at (core->io, arg, (ut8 *)str, sizeof (str) - 1);
				r_str_filter (str, strlen (str));
				res = r_str_appendf (res, "\"%s\"", str);
				break;
			case 'Z': {
				//TODO replace the hardcoded CC with the sdb ones
				ut64 len = r_debug_arg_get (core->dbg, R_ANAL_CC_TYPE_FASTCALL, i + 2);
				len = R_MIN (len + 1, sizeof (str) - 1);
				if (len == 0) {
					len = 16; // override default
				}
				(void)r_io_read_at (core->io, arg, (ut8 *)str, len);
				str[len] = 0;
				r_str_filter (str, -1);
				res = r_str_appendf (res, "\"%s\"", str);
			} break;
			default:
				res = r_str_appendf (res, "0x%08" PFMT64x "", arg);
				break;
			}
		} else {
			res = r_str_appendf (res, "0x%08" PFMT64x "", arg);
		}
		if (i + 1 < item->args) {
			res = r_str_appendf (res, ", ");
		}
	}
	r_syscall_item_free (item);
	res = r_str_appendf (res, ")");
	return res;
}

static void cmd_syscall_do(RCore *core, int n) {
	char *msg = cmd_syscall_dostr (core, n);
	if (msg) {
		r_cons_println (msg);
		free (msg);
	}
}

static void core_anal_bytes(RCore *core, const ut8 *buf, int len, int nops, int fmt) {
	int stacksize = r_config_get_i (core->config, "esil.stack.depth");
	bool iotrap = r_config_get_i (core->config, "esil.iotrap");
	bool romem = r_config_get_i (core->config, "esil.romem");
	bool stats = r_config_get_i (core->config, "esil.stats");
	bool be = core->print->big_endian;
	bool use_color = core->print->flags & R_PRINT_FLAGS_COLOR;
	core->parser->relsub = r_config_get_i (core->config, "asm.relsub");
	int ret, i, j, idx, size;
	const char *color = "";
	const char *esilstr;
	const char *opexstr;
	RAnalHint *hint;
	RAnalEsil *esil = NULL;
	RAsmOp asmop;
	RAnalOp op;
	ut64 addr;
	bool isFirst = true;
	unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
	int totalsize = 0;

	// Variables required for setting up ESIL to REIL conversion
	if (use_color) {
		color = core->cons->pal.label;
	}
	switch (fmt) {
	case 'j':
		r_cons_printf ("[");
		break;
	case 'r':
		// Setup for ESIL to REIL conversion
		esil = r_anal_esil_new (stacksize, iotrap, addrsize);
		if (!esil) {
			return;
		}
		r_anal_esil_to_reil_setup (esil, core->anal, romem, stats);
		r_anal_esil_set_pc (esil, core->offset);
		break;
	}
	for (i = idx = ret = 0; idx < len && (!nops || (nops && i < nops)); i++, idx += ret) {
		addr = core->offset + idx;
		// TODO: use more anal hints
		hint = r_anal_hint_get (core->anal, addr);
		r_asm_set_pc (core->assembler, addr);
		ret = r_asm_disassemble (core->assembler, &asmop, buf + idx, len - idx);
		ret = r_anal_op (core->anal, &op, core->offset + idx, buf + idx, len - idx, R_ANAL_OP_MASK_ALL);
		esilstr = R_STRBUF_SAFEGET (&op.esil);
		opexstr = R_STRBUF_SAFEGET (&op.opex);
		char *mnem = strdup (asmop.buf_asm);
		char *sp = strchr (mnem, ' ');
		if (sp) {
			*sp = 0;
		}
		if (ret < 1 && fmt != 'd') {
			eprintf ("Oops at 0x%08" PFMT64x " (", core->offset + idx);
			for (i = idx, j = 0; i < core->blocksize && j < 3; ++i, ++j) {
				eprintf ("%02x ", buf[i]);
			}
			eprintf ("...)\n");
			break;
		}
		size = (hint && hint->size)? hint->size: op.size;
		if (fmt == 'd') {
			char *opname = strdup (asmop.buf_asm);
			r_str_split (opname, ' ');
			char *d = r_asm_describe (core->assembler, opname);
			if (d && *d) {
				r_cons_printf ("%s: %s\n", opname, d);
				free (d);
			} else {
				eprintf ("Unknown opcode\n");
			}
			free (opname);
		} else if (fmt == 'e') {
			if (*esilstr) {
				if (use_color) {
					r_cons_printf ("%s0x%" PFMT64x Color_RESET " %s\n", color, core->offset + idx, esilstr);
				} else {
					r_cons_printf ("0x%" PFMT64x " %s\n", core->offset + idx, esilstr);
				}
			}
		} else if (fmt == 's') {
			totalsize += op.size;
		} else if (fmt == 'r') {
			if (*esilstr) {
				if (use_color) {
					r_cons_printf ("%s0x%" PFMT64x Color_RESET "\n", color, core->offset + idx);
				} else {
					r_cons_printf ("0x%" PFMT64x "\n", core->offset + idx);
				}
				r_anal_esil_parse (esil, esilstr);
				r_anal_esil_dumpstack (esil);
				r_anal_esil_stack_free (esil);
			}
		} else if (fmt == 'j') {
			if (isFirst) {
				isFirst = false;
			} else {
				r_cons_print (",");
			}
			r_cons_printf ("{\"opcode\":\"%s\",", asmop.buf_asm);
			{
				char strsub[128] = { 0 };
				// pc+33
				r_parse_varsub (core->parser, NULL,
					core->offset + idx,
					asmop.size, asmop.buf_asm,
					strsub, sizeof (strsub));
				{
					ut64 killme = UT64_MAX;
					if (r_io_read_i (core->io, op.ptr, &killme, op.refptr, be)) {
						core->parser->relsub_addr = killme;
					}
				}
				// 0x33->sym.xx
				char *p = strdup (strsub);
				r_parse_filter (core->parser, core->flags, p,
						strsub, sizeof (strsub), be);
				free (p);
				r_cons_printf ("\"disasm\":\"%s\",", strsub);
			}
			r_cons_printf ("\"mnemonic\":\"%s\",", mnem);
			if (hint && hint->opcode) {
				r_cons_printf ("\"ophint\":\"%s\",", hint->opcode);
			}
			r_cons_printf ("\"prefix\":%" PFMT64d ",", op.prefix);
			r_cons_printf ("\"id\":%d,", op.id);
			if (opexstr && *opexstr) {
				r_cons_printf ("\"opex\":%s,", opexstr);
			}
			r_cons_printf ("\"addr\":%" PFMT64d ",", core->offset + idx);
			r_cons_printf ("\"bytes\":\"");
			for (j = 0; j < size; j++) {
				r_cons_printf ("%02x", buf[j + idx]);
			}
			r_cons_printf ("\",");
			if (op.val != UT64_MAX) {
				r_cons_printf ("\"val\": %" PFMT64d ",", op.val);
			}
			if (op.ptr != UT64_MAX) {
				r_cons_printf ("\"ptr\": %" PFMT64d ",", op.ptr);
			}
			r_cons_printf ("\"size\": %d,", size);
			r_cons_printf ("\"type\": \"%s\",",
				r_anal_optype_to_string (op.type));
			if (op.reg) {
				r_cons_printf ("\"reg\": \"%s\",", op.reg);
			}
			if (hint && hint->esil) {
				r_cons_printf ("\"esil\": \"%s\",", hint->esil);
			} else if (*esilstr) {
				r_cons_printf ("\"esil\": \"%s\",", esilstr);
			}
			if (hint && hint->jump != UT64_MAX) {
				op.jump = hint->jump;
			}
			if (op.jump != UT64_MAX) {
				r_cons_printf ("\"jump\":%" PFMT64d ",", op.jump);
			}
			if (hint && hint->fail != UT64_MAX) {
				op.fail = hint->fail;
			}
			if (op.refptr != -1) {
				r_cons_printf ("\"refptr\":%d,", op.refptr);
			}
			if (op.fail != UT64_MAX) {
				r_cons_printf ("\"fail\":%" PFMT64d ",", op.fail);
			}
			r_cons_printf ("\"cycles\":%d,", op.cycles);
			if (op.failcycles) {
				r_cons_printf ("\"failcycles\":%d,", op.failcycles);
			}
			r_cons_printf ("\"delay\":%d,", op.delay);
			{
				const char *p = r_anal_stackop_tostring (op.stackop);
				if (p && *p && strcmp (p, "null"))
					r_cons_printf ("\"stack\":\"%s\",", p);
			}
			if (op.stackptr) {
				r_cons_printf ("\"stackptr\":%d,", op.stackptr);
			}
			{
				const char *arg = (op.type & R_ANAL_OP_TYPE_COND)
					? r_anal_cond_tostring (op.cond): NULL;
				if (arg) {
					r_cons_printf ("\"cond\":\"%s\",", arg);
				}
			}
			r_cons_printf ("\"family\":\"%s\"}", r_anal_op_family_to_string (op.family));
		} else {
#define printline(k, fmt, arg)\
	{ \
		if (use_color)\
			r_cons_printf ("%s%s: " Color_RESET, color, k);\
		else\
			r_cons_printf ("%s: ", k);\
		if (fmt) r_cons_printf (fmt, arg);\
	}
			printline ("address", "0x%" PFMT64x "\n", core->offset + idx);
			printline ("opcode", "%s\n", asmop.buf_asm);
			printline ("mnemonic", "%s\n", mnem);
			if (hint) {
				if (hint->opcode) {
					printline ("ophint", "%s\n", hint->opcode);
				}
#if 0
				// addr should not override core->offset + idx.. its silly
				if (hint->addr != UT64_MAX) {
					printline ("addr", "0x%08" PFMT64x "\n", (hint->addr + idx));
				}
#endif
			}
			printline ("prefix", "%" PFMT64d "\n", op.prefix);
			printline ("id", "%d\n", op.id);
#if 0
// no opex here to avoid lot of tests broken..and having json in here is not much useful imho
			if (opexstr && *opexstr) {
				printline ("opex", "%s\n", opexstr);
			}
#endif
			printline ("bytes", NULL, 0);
			for (j = 0; j < size; j++) {
				r_cons_printf ("%02x", buf[j + idx]);
			}
			r_cons_newline ();
			if (op.val != UT64_MAX)
				printline ("val", "0x%08" PFMT64x "\n", op.val);
			if (op.ptr != UT64_MAX)
				printline ("ptr", "0x%08" PFMT64x "\n", op.ptr);
			if (op.refptr != -1)
				printline ("refptr", "%d\n", op.refptr);
			printline ("size", "%d\n", size);
			printline ("type", "%s\n", r_anal_optype_to_string (op.type));
			{
				const char *t2 = r_anal_optype_to_string (op.type2);
				if (t2 && strcmp (t2, "null")) {
					printline ("type2", "%s\n", t2);
				}
			}
			if (op.reg) {
				printline ("reg", "%s\n", op.reg);
			}
			if (hint && hint->esil) {
				printline ("esil", "%s\n", hint->esil);
			} else if (*esilstr) {
				printline ("esil", "%s\n", esilstr);
			}
			if (hint && hint->jump != UT64_MAX) {
				op.jump = hint->jump;
			}
			if (op.jump != UT64_MAX) {
				printline ("jump", "0x%08" PFMT64x "\n", op.jump);
			}
			if (op.direction != 0) {
				const char * dir = op.direction == 1 ? "read"
					: op.direction == 2 ? "write"
					: op.direction == 4 ? "exec"
					: op.direction == 8 ? "ref": "none";
				printline ("direction", "%s\n", dir);
			}
			if (hint && hint->fail != UT64_MAX) {
				op.fail = hint->fail;
			}
			if (op.fail != UT64_MAX) {
				printline ("fail", "0x%08" PFMT64x "\n", op.fail);
			}
			if (op.delay) {
				printline ("delay", "%d\n", op.delay);
			}
			printline ("stack", "%s\n", r_anal_stackop_tostring (op.stackop));
			{
				const char *arg = (op.type & R_ANAL_OP_TYPE_COND)?  r_anal_cond_tostring (op.cond): NULL;
				if (arg) {
					printline ("cond", "%s\n", arg);
				}
			}
			printline ("family", "%s\n", r_anal_op_family_to_string (op.family));
			printline ("stackop", "%s\n", r_anal_stackop_tostring (op.stackop));
			if (op.stackptr) {
				printline ("stackptr", "%"PFMT64d"\n", op.stackptr);
			}
		}
		//r_cons_printf ("false: 0x%08"PFMT64x"\n", core->offset+idx);
		//free (hint);
		free (mnem);
		r_anal_hint_free (hint);
	}
	if (fmt == 'j') {
		r_cons_printf ("]");
		r_cons_newline ();
	} else if (fmt == 's') {
		r_cons_printf ("%d\n", totalsize);
	}
	r_anal_esil_free (esil);
}

static int bb_cmp(const void *a, const void *b) {
	const RAnalBlock *ba = a;
	const RAnalBlock *bb = b;
	return ba->addr - bb->addr;
}

static int anal_fcn_list_bb(RCore *core, const char *input, bool one) {
	RDebugTracepoint *tp = NULL;
	RListIter *iter;
	RAnalBlock *b;
	int mode = 0;
	ut64 addr, bbaddr = UT64_MAX;
	bool firstItem = true;

	if (*input == '.') {
		one = true;
		input++;
	}
	if (*input) {
		mode = *input;
		input++;
	}
	if (*input == '.') {
		one = true;
		input++;
	}
	if (input && *input) {
		addr = bbaddr = r_num_math (core->num, input);
	} else {
		addr = core->offset;
	}
	if (one) {
		bbaddr = addr;
	}
	RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, 0);
	if (!fcn) {
		return false;
	}
	switch (mode) {
	case 'j':
		r_cons_printf ("[");
		break;
	case '*':
		r_cons_printf ("fs blocks\n");
		break;
	}
	r_list_sort (fcn->bbs, bb_cmp);
	r_list_foreach (fcn->bbs, iter, b) {
		if (one) {
			if (bbaddr != UT64_MAX && (bbaddr < b->addr || bbaddr >= (b->addr + b->size))) {
				continue;
			}
		}
		switch (mode) {
		case 'r':
			if (b->jump == UT64_MAX) {
				ut64 retaddr = b->addr;
				if (b->op_pos) {
					retaddr += b->op_pos[b->ninstr - 2];
				}
				if (!strcmp (input, "*")) {
					r_cons_printf ("db 0x%08"PFMT64x"\n", retaddr);
				} else if (!strcmp (input, "-*")) {
					r_cons_printf ("db-0x%08"PFMT64x"\n", retaddr);
				} else {
					r_cons_printf ("0x%08"PFMT64x"\n", retaddr);
				}
			}
			break;
		case '*':
			r_cons_printf ("f bb.%05" PFMT64x " = 0x%08" PFMT64x "\n",
				b->addr & 0xFFFFF, b->addr);
			break;
		case 'q':
			r_cons_printf ("0x%08" PFMT64x "\n", b->addr);
			break;
		case 'j':
			//r_cons_printf ("%" PFMT64d "%s", b->addr, iter->n? ",": "");
			{
			RListIter *iter2;
			RAnalBlock *b2;
			int inputs = 0;
			int outputs = 0;
			r_list_foreach (fcn->bbs, iter2, b2) {
				if (b2->jump == b->addr) {
					inputs++;
				}
				if (b2->fail == b->addr) {
					inputs++;
				}
			}
			if (b->jump != UT64_MAX) {
				outputs ++;
			}
			if (b->fail != UT64_MAX) {
				outputs ++;
			}
			r_cons_printf ("%s{", firstItem? "": ",");
			firstItem = false;
			if (b->jump != UT64_MAX) {
				r_cons_printf ("\"jump\":%"PFMT64d",", b->jump);
			}
			if (b->fail != UT64_MAX) {
				r_cons_printf ("\"fail\":%"PFMT64d",", b->fail);
			}
			r_cons_printf ("\"addr\":%" PFMT64d ",\"size\":%d,\"inputs\":%d,\"outputs\":%d,\"ninstr\":%d,\"traced\":%s}",
				b->addr, b->size, inputs, outputs, b->ninstr, r_str_bool (b->traced));
			}
			break;
		case 'i':
			{
			RListIter *iter2;
			RAnalBlock *b2;
			int inputs = 0;
			int outputs = 0;
			r_list_foreach (fcn->bbs, iter2, b2) {
				if (b2->jump == b->addr) {
					inputs++;
				}
				if (b2->fail == b->addr) {
					inputs++;
				}
			}
			if (b->jump != UT64_MAX) {
				outputs ++;
			}
			if (b->fail != UT64_MAX) {
				outputs ++;
			}
			firstItem = false;
			if (b->jump != UT64_MAX) {
				r_cons_printf ("jump: 0x%08"PFMT64x"\n", b->jump);
			}
			if (b->fail != UT64_MAX) {
				r_cons_printf ("fail: 0x%08"PFMT64x"\n", b->fail);
			}
			r_cons_printf ("addr: 0x%08"PFMT64x"\nsize: %d\ninputs: %d\noutputs: %d\nninstr: %d\ntraced: %s\n",
				b->addr, b->size, inputs, outputs, b->ninstr, r_str_bool (b->traced));
			}
			break;
		default:
			tp = r_debug_trace_get (core->dbg, b->addr);
			r_cons_printf ("0x%08" PFMT64x " 0x%08" PFMT64x " %02X:%04X %d",
				b->addr, b->addr + b->size,
				tp? tp->times: 0, tp? tp->count: 0,
				b->size);
			if (b->jump != UT64_MAX) {
				r_cons_printf (" j 0x%08" PFMT64x, b->jump);
			}
			if (b->fail != UT64_MAX) {
				r_cons_printf (" f 0x%08" PFMT64x, b->fail);
			}
			r_cons_newline ();
			break;
		}
	}
	if (mode == 'j') {
		r_cons_printf ("]\n");
	}
	return true;
}

static bool anal_bb_edge (RCore *core, const char *input) {
	// "afbe" switch-bb-addr case-bb-addr
	char *arg = strdup (r_str_trim_ro(input));
	char *sp = strchr (arg, ' ');
	if (sp) {
		*sp++ = 0;
		ut64 sw_at = r_num_math (core->num, arg);
		ut64 cs_at = r_num_math (core->num, sp);
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, sw_at, 0);
		if (fcn) {
			RAnalBlock *bb;
			RListIter *iter;
			r_list_foreach (fcn->bbs, iter, bb) {
				if (sw_at >= bb->addr && sw_at < (bb->addr + bb->size)) {
					if (!bb->switch_op) {
						bb->switch_op = r_anal_switch_op_new (
							sw_at, 0, 0);
					}
					r_anal_switch_op_add_case (bb->switch_op, cs_at, 0, cs_at);
				}
			}
			free (arg);
			return true;
		}
	}
	free (arg);
	return false;
}

static bool anal_fcn_del_bb(RCore *core, const char *input) {
	ut64 addr = r_num_math (core->num, input);
	if (!addr) {
		addr = core->offset;
	}
	RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, -1);
	if (fcn) {
		if (!strcmp (input, "*")) {
			r_list_free (fcn->bbs);
			fcn->bbs = NULL;
		} else {
			RAnalBlock *b;
			RListIter *iter;
			r_list_foreach (fcn->bbs, iter, b) {
				if (b->addr == addr) {
					r_list_delete (fcn->bbs, iter);
					return true;
				}
			}
			eprintf ("Cannot find basic block\n");
		}
	} else {
		eprintf ("Cannot find function\n");
	}
	return false;
}

static int anal_fcn_add_bb(RCore *core, const char *input) {
	// fcn_addr bb_addr bb_size [jump] [fail]
	char *ptr;
	const char *ptr2 = NULL;
	ut64 fcnaddr = -1LL, addr = -1LL;
	ut64 size = 0LL;
	ut64 jump = UT64_MAX;
	ut64 fail = UT64_MAX;
	int type = R_ANAL_BB_TYPE_NULL;
	RAnalFunction *fcn = NULL;
	RAnalDiff *diff = NULL;

	while (*input == ' ') input++;
	ptr = strdup (input);

	switch (r_str_word_set0 (ptr)) {
	case 7:
		ptr2 = r_str_word_get0 (ptr, 6);
		if (!(diff = r_anal_diff_new ())) {
			eprintf ("error: Cannot init RAnalDiff\n");
			free (ptr);
			return false;
		}
		if (ptr2[0] == 'm') {
			diff->type = R_ANAL_DIFF_TYPE_MATCH;
		} else if (ptr2[0] == 'u') {
			diff->type = R_ANAL_DIFF_TYPE_UNMATCH;
		}
	case 6:
		ptr2 = r_str_word_get0 (ptr, 5);
		if (strchr (ptr2, 'h')) {
			type |= R_ANAL_BB_TYPE_HEAD;
		}
		if (strchr (ptr2, 'b')) {
			type |= R_ANAL_BB_TYPE_BODY;
		}
		if (strchr (ptr2, 'l')) {
			type |= R_ANAL_BB_TYPE_LAST;
		}
		if (strchr (ptr2, 'f')) {
			type |= R_ANAL_BB_TYPE_FOOT;
		}
	case 5: // get fail
		fail = r_num_math (core->num, r_str_word_get0 (ptr, 4));
	case 4: // get jump
		jump = r_num_math (core->num, r_str_word_get0 (ptr, 3));
	case 3: // get size
		size = r_num_math (core->num, r_str_word_get0 (ptr, 2));
	case 2: // get addr
		addr = r_num_math (core->num, r_str_word_get0 (ptr, 1));
	case 1: // get fcnaddr
		fcnaddr = r_num_math (core->num, r_str_word_get0 (ptr, 0));
	}
	fcn = r_anal_get_fcn_in (core->anal, fcnaddr, 0);
	if (fcn) {
		int ret = r_anal_fcn_add_bb (core->anal, fcn, addr, size, jump, fail, type, diff);
		if (!ret) {
			eprintf ("Cannot add basic block\n");
		}
	} else {
		eprintf ("Cannot find function at 0x%" PFMT64x "\n", fcnaddr);
	}
	r_anal_diff_free (diff);
	free (ptr);
	return true;
}

static void r_core_anal_nofunclist  (RCore *core, const char *input) {
	int minlen = (int)(input[0]==' ') ? r_num_math (core->num, input + 1): 16;
	ut64 code_size = r_num_get (core->num, "$SS");
	ut64 base_addr = r_num_get (core->num, "$S");
	ut64 chunk_size, chunk_offset, i;
	RListIter *iter, *iter2;
	RAnalFunction *fcn;
	RAnalBlock *b;
	char* bitmap;
	int counter;

	if (minlen < 1) {
		minlen = 1;
	}
	if (code_size < 1) {
		return;
	}
	bitmap = calloc (1, code_size+64);
	if (!bitmap) {
		return;
	}

	// for each function
	r_list_foreach (core->anal->fcns, iter, fcn) {
		// for each basic block in the function
		r_list_foreach (fcn->bbs, iter2, b) {
			// if it is not withing range, continue
			if ((fcn->addr < base_addr) || (fcn->addr >= base_addr+code_size))
				continue;
			// otherwise mark each byte in the BB in the bitmap
			for (counter = 0; counter < b->size; counter++) {
				bitmap[b->addr+counter-base_addr] = '=';
			}
			// finally, add a special marker to show the beginning of a
			// function
			bitmap[fcn->addr-base_addr] = 'F';
		}
	}

	// Now we print the list of memory regions that are not assigned to a function
	chunk_size = 0;
	chunk_offset = 0;
	for (i = 0; i < code_size; i++) {
		if (bitmap[i]){
			// We only print a region is its size is bigger than 15 bytes
			if (chunk_size >= minlen){
				fcn = r_anal_get_fcn_in (core->anal, base_addr+chunk_offset, R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM);
				if (fcn) {
					r_cons_printf ("0x%08"PFMT64x"  %6d   %s\n", base_addr+chunk_offset, chunk_size, fcn->name);
				} else {
					r_cons_printf ("0x%08"PFMT64x"  %6d\n", base_addr+chunk_offset, chunk_size);
				}
			}
			chunk_size = 0;
			chunk_offset = i+1;
			continue;
		}
		chunk_size+=1;
	}
	if (chunk_size >= 16) {
		fcn = r_anal_get_fcn_in (core->anal, base_addr+chunk_offset, R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM);
		if (fcn) {
			r_cons_printf ("0x%08"PFMT64x"  %6d   %s\n", base_addr+chunk_offset, chunk_size, fcn->name);
		} else {
			r_cons_printf ("0x%08"PFMT64x"  %6d\n", base_addr+chunk_offset, chunk_size);
		}
	}
	free(bitmap);
}

static void r_core_anal_fmap  (RCore *core, const char *input) {
	int show_color = r_config_get_i (core->config, "scr.color");
	int cols = r_config_get_i (core->config, "hex.cols") * 4;
	ut64 code_size = r_num_get (core->num, "$SS");
	ut64 base_addr = r_num_get (core->num, "$S");
	RListIter *iter, *iter2;
	RAnalFunction *fcn;
	RAnalBlock *b;
	char* bitmap;
	int assigned;
	ut64 i;

	if (code_size < 1) {
		return;
	}
	bitmap = calloc (1, code_size+64);
	if (!bitmap) {
		return;
	}

	// for each function
	r_list_foreach (core->anal->fcns, iter, fcn) {
		// for each basic block in the function
		r_list_foreach (fcn->bbs, iter2, b) {
			// if it is not within range, continue
			if ((fcn->addr < base_addr) || (fcn->addr >= base_addr+code_size))
				continue;
			// otherwise mark each byte in the BB in the bitmap
			int counter = 1;
			for (counter = 0; counter < b->size; counter++) {
				bitmap[b->addr+counter-base_addr] = '=';
			}
			bitmap[fcn->addr-base_addr] = 'F';
		}
	}
	// print the bitmap
	assigned = 0;
	if (cols < 1) {
		cols = 1;
	}
	for (i = 0; i < code_size; i += 1) {
		if (!(i % cols)) {
			r_cons_printf ("\n0x%08"PFMT64x"  ", base_addr+i);
		}
		if (bitmap[i]) {
			assigned++;
		}
		if (show_color) {
			if (bitmap[i]) {
				r_cons_printf ("%s%c\x1b[0m", Color_GREEN, bitmap[i]);
			} else {
				r_cons_printf (".");
			}
		} else {
			r_cons_printf ("%c", bitmap[i] ? bitmap[i] : '.' );
		}
	}
	r_cons_printf ("\n%d / %d (%.2lf%%) bytes assigned to a function\n", assigned, code_size, 100.0*( (float) assigned) / code_size);
	free(bitmap);
}

static bool fcnNeedsPrefix(const char *name) {
	if (!strncmp (name, "entry", 5)) {
		return false;
	}
	if (!strncmp (name, "main", 4)) {
		return false;
	}
	return (!strchr (name, '.'));
}

/* TODO: move into r_anal_fcn_rename(); */
static bool setFunctionName(RCore *core, ut64 off, const char *_name, bool prefix) {
	char *name, *oname, *nname = NULL;
	RAnalFunction *fcn;
	if (!core || !_name) {
		return false;
	}
	const char *fcnpfx = r_config_get (core->config, "anal.fcnprefix");
	if (!fcnpfx) {
		fcnpfx = "fcn";
	}
	if (r_reg_get (core->anal->reg, _name, -1)) {
		name = r_str_newf ("%s.%s", fcnpfx, _name);
	} else {
		name = strdup (_name);
	}
	fcn = r_anal_get_fcn_in (core->anal, off,
				R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM | R_ANAL_FCN_TYPE_LOC);
	if (!fcn) {
		return false;
	}
	if (prefix && fcnNeedsPrefix (name)) {
		nname = r_str_newf ("%s.%s", fcnpfx, name);
	} else {
		nname = strdup (name);
	}
	oname = fcn->name;
	r_flag_rename (core->flags, r_flag_get (core->flags, fcn->name), nname);
	fcn->name = strdup (nname);
	if (core->anal->cb.on_fcn_rename) {
		core->anal->cb.on_fcn_rename (core->anal,
					core->anal->user, fcn, nname);
	}
	free (oname);
	free (nname);
	free (name);
	return true;
}

static void afCc(RCore *core, const char *input) {
	ut64 addr;
	RAnalFunction *fcn;
	if (*input == ' ') {
		addr = r_num_math (core->num, input);
	} else {
		addr = core->offset;
	}
	if (addr == 0LL) {
		fcn = r_anal_fcn_find_name (core->anal, input + 3);
	} else {
		fcn = r_anal_get_fcn_in (core->anal, addr, R_ANAL_FCN_TYPE_NULL);
	}
	if (fcn) {
		ut32 totalCycles = r_anal_fcn_cost (core->anal, fcn);
		// FIXME: This defeats the purpose of the function, but afC is used in project files.
		// cf. canal.c
		r_cons_printf ("%d\n", totalCycles);
	} else {
		eprintf ("Cannot find function\n");
	}
}

static int cmd_anal_fcn(RCore *core, const char *input) {
	char i;

	r_cons_break_timeout (r_config_get_i (core->config, "anal.timeout"));
	switch (input[1]) {
	case 'f': // "aff"
		r_anal_fcn_fit_overlaps (core->anal, NULL);
		break;
	case 'a':
		if (input[2] == 'l') { // afal : list function call arguments
			int show_args = r_config_get_i (core->config, "dbg.funcarg");
			if (show_args) {
				r_core_print_func_args (core);
			}
			break;
		}
	case 'd': // "afd"
		{
		ut64 addr = 0;
		if (input[2] == '?') {
			eprintf ("afd [offset]\n");
		} else if (input[2] == ' ') {
			addr = r_num_math (core->num, input + 2);
		} else {
			addr = core->offset;
		}
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, 0);
		if (fcn) {
			if (fcn->addr != addr) {
				r_cons_printf ("%s + %d\n", fcn->name,
						(int)(addr - fcn->addr));
			} else {
				r_cons_println (fcn->name);
			}
		} else {
			eprintf ("Cannot find function\n");
		}
		}
		break;
	case '-': // "af-"
		if (!input[2] || !strcmp (input + 2, "*")) {
			r_anal_fcn_del_locs (core->anal, UT64_MAX);
			r_anal_fcn_del (core->anal, UT64_MAX);
		} else {
			ut64 addr = input[2]
				? r_num_math (core->num, input + 2)
				: core->offset;
			r_anal_fcn_del_locs (core->anal, addr);
			r_anal_fcn_del (core->anal, addr);
		}
		break;
	case 'u': // "afu"
		{
		ut64 addr = core->offset;
		ut64 addr_end = r_num_math (core->num, input + 2);
		if (addr_end < addr) {
			eprintf ("Invalid address ranges\n");
		} else {
			int depth = 1;
			ut64 a, b;
			const char *c;
			a = r_config_get_i (core->config, "anal.from");
			b = r_config_get_i (core->config, "anal.to");
			c = r_config_get (core->config, "anal.limits");
			r_config_set_i (core->config, "anal.from", addr);
			r_config_set_i (core->config, "anal.to", addr_end);
			r_config_set (core->config, "anal.limits", "true");

			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, 0);
			if (fcn) {
				r_anal_fcn_resize (core->anal, fcn, addr_end - addr);
			}
			r_core_anal_fcn (core, addr, UT64_MAX,
					R_ANAL_REF_TYPE_NULL, depth);
			fcn = r_anal_get_fcn_in (core->anal, addr, 0);
			if (fcn) {
				r_anal_fcn_resize (core->anal, fcn, addr_end - addr);
			}
			r_config_set_i (core->config, "anal.from", a);
			r_config_set_i (core->config, "anal.to", b);
			r_config_set (core->config, "anal.limits", c? c: "");
		}
		}
		break;
	case '+': { // "af+"
		if (input[2] != ' ') {
			eprintf ("Missing arguments\n");
			return false;
		}
		char *ptr = strdup (input + 3);
		const char *ptr2;
		int n = r_str_word_set0 (ptr);
		const char *name = NULL;
		ut64 addr = UT64_MAX;
		ut64 size = 0LL;
		RAnalDiff *diff = NULL;
		int type = R_ANAL_FCN_TYPE_FCN;
		if (n > 1) {
			switch (n) {
			case 5:
				size = r_num_math (core->num, r_str_word_get0 (ptr, 4));
			case 4:
				ptr2 = r_str_word_get0 (ptr, 3);
				if (!(diff = r_anal_diff_new ())) {
					eprintf ("error: Cannot init RAnalDiff\n");
					free (ptr);
					return false;
				}
				if (ptr2[0] == 'm') {
					diff->type = R_ANAL_DIFF_TYPE_MATCH;
				} else if (ptr2[0] == 'u') {
					diff->type = R_ANAL_DIFF_TYPE_UNMATCH;
				}
			case 3:
				ptr2 = r_str_word_get0 (ptr, 2);
				if (strchr (ptr2, 'l')) {
					type = R_ANAL_FCN_TYPE_LOC;
				} else if (strchr (ptr2, 'i')) {
					type = R_ANAL_FCN_TYPE_IMP;
				} else if (strchr (ptr2, 's')) {
					type = R_ANAL_FCN_TYPE_SYM;
				} else {
					type = R_ANAL_FCN_TYPE_FCN;
				}
			case 2:
				name = r_str_word_get0 (ptr, 1);
			case 1:
				addr = r_num_math (core->num, r_str_word_get0 (ptr, 0));
			}
			if (!r_anal_fcn_add (core->anal, addr, size, name, type, diff)) {
				eprintf ("Cannot add function (duplicated)\n");
			}
		}
		r_anal_diff_free (diff);
		free (ptr);
		}
		break;
	case 'o': // "afo"
		{
		RAnalFunction *fcn;
		ut64 addr = core->offset;
		if (input[2] == ' ')
			addr = r_num_math (core->num, input + 3);
		if (addr == 0LL) {
			fcn = r_anal_fcn_find_name (core->anal, input + 3);
		} else {
			fcn = r_anal_get_fcn_in (core->anal, addr, R_ANAL_FCN_TYPE_NULL);
		}
		if (fcn) {
			r_cons_printf ("0x%08" PFMT64x "\n", fcn->addr);
		}
		}
		break;
	case 'i': // "afi"
		switch (input[2]) {
		case '?':
			r_core_cmd_help (core, help_msg_afi);
			break;
		case '.': // "afi."
			{
				ut64 addr = core->offset;
				if (input[3] == ' ') {
					addr = r_num_math (core->num, input + 3);
				}
				RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, R_ANAL_FCN_TYPE_NULL);
				if (fcn) {
					r_cons_printf ("%s\n", fcn->name);
				}
			}
			break;
		case 'l': // "afil"
			if (input[3] == '?') {
				// TODO #7967 help refactor
				help_msg_afll[1] = "afil";
				r_core_cmd_help (core, help_msg_afll);
				break;
			}
			/* fallthrough */
		case 'j':   // "afij"
		case '*':   // "afi*"
			r_core_anal_fcn_list (core, input + 3, input + 2);
			break;
		default:
			i = 1;
			r_core_anal_fcn_list (core, input + 2, &i);
			break;
		}
		break;
	case 'l': // "afl"
		switch (input[2]) {
		case '?':
			r_core_cmd_help (core, help_msg_afl);
			break;
		case 'l': // "afll"
			if (input[3] == '?') {
				// TODO #7967 help refactor
				help_msg_afll[1] = "afll";
				r_core_cmd_help (core, help_msg_afll);
				break;
			}
			/* fallthrough */
		case 'j': // "aflj"
		case 'q': // "aflq"
		case 's': // "afls"
		case '*': // "afl*"
			r_core_anal_fcn_list (core, NULL, input + 2);
			break;
		case 'c': // "aflc"
			r_cons_printf ("%d\n", r_list_length (core->anal->fcns));
			break;
		default: // "afl "
			r_core_anal_fcn_list (core, NULL, "o");
			break;
		}
		break;
	case 's': // "afs"
		{
		ut64 addr;
		RAnalFunction *f;
		const char *arg = input + 3;
		if (input[2] && (addr = r_num_math (core->num, arg))) {
			arg = strchr (arg, ' ');
			if (arg) {
				arg++;
			}
		} else {
			addr = core->offset;
		}
		if ((f = r_anal_get_fcn_in (core->anal, addr, R_ANAL_FCN_TYPE_NULL))) {
			if (arg && *arg) {
				r_anal_str_to_fcn (core->anal, f, arg);
			} else {
				char *str = r_anal_fcn_to_string (core->anal, f);
				r_cons_println (str);
				free (str);
			}
		} else {
			eprintf ("No function defined at 0x%08" PFMT64x "\n", addr);
		}
		}
		break;
	case 'm': // "afm" - merge two functions
		r_core_anal_fcn_merge (core, core->offset, r_num_math (core->num, input + 2));
		break;
	case 'M': // "afM" - print functions map
		r_core_anal_fmap (core, input + 1);
		break;
	case 'v': // "afv"
		var_cmd (core, input + 2);
		break;
	case 't': // "aft"
		type_cmd (core, input + 2);
		break;
	case 'C': // "afC"
		if (input[2] == 'c') {
			RAnalFunction *fcn;
			if ((fcn = r_anal_get_fcn_in (core->anal, core->offset, 0)) != NULL) {
				r_cons_printf ("%i\n", r_anal_fcn_cc (fcn));
			} else {
				eprintf ("Error: Cannot find function at 0x08%" PFMT64x "\n", core->offset);
			}
		} else if (input[2] == 'l') {
			RAnalFunction *fcn;
			if ((fcn = r_anal_get_fcn_in (core->anal, core->offset, 0)) != NULL) {
				r_cons_printf ("%d\n", r_anal_fcn_loops (fcn));
			} else {
				eprintf ("Error: Cannot find function at 0x08%" PFMT64x "\n", core->offset);
			}
		} else if (input[2] == '?') {
			r_core_cmd_help (core, help_msg_afC);
		} else {
			afCc (core, input + 3);
		}
		break;
	case 'c':{ // "afc"
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, 0);
		if (!fcn && !(input[2] == '?'|| input[2] == 'l' || input[2] == 'o')) {
			eprintf ("Cannot find function here\n");
			break;
		}
		switch (input[2]) {
		case '\0': // "afc"
			r_cons_println (fcn->cc);
			break;
		case ' ': { // "afc "
			char *cc = r_str_trim (strdup (input + 3));
			if (!r_anal_cc_exist (core->anal, cc)) {
				eprintf ("Unknown calling convention '%s'\n"
						"See afcl for available types\n", cc);
			} else {
				fcn->cc = r_str_const (r_anal_cc_to_constant (core->anal, cc));
			}
			break;
		}
		case 'a': // "afca""
			eprintf ("Todo\n");
			break;
		case 'l': // "afcl" list all function Calling conventions.
			sdb_foreach (core->anal->sdb_cc, cc_print, NULL);
			break;
		case 'o': { // "afco"
			char *dbpath = r_str_trim (strdup (input + 3));
			if (r_file_exists (dbpath)) {
				Sdb *db = sdb_new (0, dbpath, 0);
				sdb_merge (core->anal->sdb_cc, db);
				sdb_close (db);
				sdb_free (db);
			}
			free (dbpath);
			break;
		}
		case 'r': {	// "afcr"
			int i;
			char *out, *cmd, *regname, *tmp;
			char *subvec_str = r_str_new ("");
			char *json_str = r_str_new ("");
			// if json_str initialize to NULL, it's possible for afcrj to output a (NULL)
			// subvec_str and json_str should be valid until exiting this code block
			bool json = input[3] == 'j'? true: false;
			for (i = 0; i <= 11; i++) {
				if (i == 0) {
					cmd = r_str_newf ("cc.%s.ret", fcn->cc);
				} else {
					cmd = r_str_newf ("cc.%s.arg%d", fcn->cc, i);
				}
				if (i < 7) {
					regname = r_str_new (cmd);
				} else {
					regname = r_str_newf ("cc.%s.float_arg%d", fcn->cc, i - 6);
				}
				out = sdb_querys (core->anal->sdb_cc, NULL, 0, cmd);
				free (cmd);
				if (out) {
					out[strlen (out) - 1] = 0;
					if (json) {
						tmp = subvec_str;
						subvec_str = r_str_newf ("%s,\"%s\"", subvec_str, out);
						free (tmp);
					} else {
						r_cons_printf ("%s: %s\n", regname, out);
					}
					free (out);
				}
				free (regname);
				if (!subvec_str[0]) {
					continue;
				}
				switch (i) {
				case 0: {
					tmp = json_str;
					json_str = r_str_newf ("%s,\"ret\":%s", json_str, subvec_str + 1);
					free (tmp);
				} break;
				case 6: {
					tmp = json_str;
					json_str = r_str_newf ("%s,\"args\":[%s]", json_str, subvec_str + 1);
					free (tmp);
				} break;
				case 11: {
					tmp = json_str;
					json_str = r_str_newf ("%s,\"float_args\":[%s]", json_str, subvec_str + 1);
					free (tmp);
				} break;
				default:
					continue;
				}
				free (subvec_str);
				subvec_str = r_str_new ("");
			}
			if (json && json_str[0]) {
				r_cons_printf ("{%s}\n", json_str + 1);
			}
			free (subvec_str);
			free (json_str);
		} break;
		case '?': // "afc?"
		default:
			r_core_cmd_help (core, help_msg_afc);
		}
		}break;
	case 'B': // "afB" // set function bits
		if (input[2] == ' ') {
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset,
					R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM);
			if (fcn) {
				int bits = atoi (input + 3);
				r_anal_hint_set_bits (core->anal, fcn->addr, bits);
				r_anal_hint_set_bits (core->anal,
					fcn->addr + r_anal_fcn_size (fcn),
					core->anal->bits);
				fcn->bits = bits;
			} else {
				eprintf ("Cannot find function to set bits\n");
			}
		} else {
			eprintf ("Usage: afB [bits]\n");
		}
		break;
	case 'b': // "afb"
		switch (input[2]) {
		case '-': // "afb-"
			anal_fcn_del_bb (core, input + 3);
			break;
		case 'e': // "afbe"
			anal_bb_edge (core, input + 3);
			break;
		case 0:
		case ' ': // "afb "
		case 'q': // "afbq"
		case 'r': // "afbr"
		case '*': // "afb*"
		case 'j': // "afbj"
			anal_fcn_list_bb (core, input + 2, false);
			break;
		case 'i': // "afbi"
			anal_fcn_list_bb (core, input + 2, true);
			break;
		case '.': // "afb."
			anal_fcn_list_bb (core, input[2]? " $$": input + 2, true);
			break;
		case '+': // "afb+"
			anal_fcn_add_bb (core, input + 3);
			break;
		default:
		case '?':
			r_core_cmd_help (core, help_msg_afb);
			break;
		}
		break;
	case 'n': // "afn"
		switch (input[2]) {
		case 's': // "afns"
			free (r_core_anal_fcn_autoname (core, core->offset, 1));
			break;
		case 'a': // "afna"
			{
			char *name = r_core_anal_fcn_autoname (core, core->offset, 0);
			if (name) {
				r_cons_printf ("afn %s 0x%08" PFMT64x "\n", name, core->offset);
				free (name);
			}
			}
			break;
		case 0: // "afn"
			{
				RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
				if (fcn) {
					r_cons_printf ("%s\n", fcn->name);
				}
			}
			break;
		case ' ': // "afn "
			{
			ut64 off = core->offset;
			char *p, *name = strdup (input + 3);
			if ((p = strchr (name, ' '))) {
				*p++ = 0;
				off = r_num_math (core->num, p);
			}
			if (*name) {
				if (!setFunctionName (core, off, name, false)) {
					eprintf ("Cannot find function '%s' at 0x%08" PFMT64x "\n", name, off);
				}
				free (name);
			} else {
				eprintf ("Usage: afn newname [off]   # set new name to given function\n");
				free (name);
			}
			}
			break;
		default:
			r_core_cmd_help (core, help_msg_afn);
			break;
		}
		break;
	case 'S': // afS"
		{
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
		if (fcn) {
			fcn->maxstack = r_num_math (core->num, input + 3);
			//fcn->stack = fcn->maxstack;
		}
		}
		break;
#if 0
	/* this is undocumented and probably have no uses. plz discuss */
	case 'e': // "afe"
		{
		RAnalFunction *fcn;
		ut64 off = core->offset;
		char *p, *name = strdup ((input[2]&&input[3])? input + 3: "");
		if ((p = strchr (name, ' '))) {
			*p = 0;
			off = r_num_math (core->num, p + 1);
		}
		fcn = r_anal_get_fcn_in (core->anal, off, R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM);
		if (fcn) {
			RAnalBlock *b;
			RListIter *iter;
			RAnalRef *r;
			r_list_foreach (fcn->refs, iter, r) {
				r_cons_printf ("0x%08" PFMT64x " -%c 0x%08" PFMT64x "\n", r->at, r->type, r->addr);
			}
			r_list_foreach (fcn->bbs, iter, b) {
				int ok = 0;
				if (b->type == R_ANAL_BB_TYPE_LAST) ok = 1;
				if (b->type == R_ANAL_BB_TYPE_FOOT) ok = 1;
				if (b->jump == UT64_MAX && b->fail == UT64_MAX) ok = 1;
				if (ok) {
					r_cons_printf ("0x%08" PFMT64x " -r\n", b->addr);
					// TODO: check if destination is outside the function boundaries
				}
			}
		} else eprintf ("Cannot find function at 0x%08" PFMT64x "\n", core->offset);
		free (name);
		}
		break;
#endif
	case 'x': // "afx"
		switch (input[2]) {
		case '\0': // "afx"
		case 'j': // "afxj"
		case ' ': // "afx "
#if FCN_OLD
			if (input[2] == 'j') {
				r_cons_printf ("[");
			}
			// TODO: sdbize!
			// list xrefs from current address
			{
				ut64 addr = input[2]==' '? r_num_math (core->num, input + 2): core->offset;
				RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, R_ANAL_FCN_TYPE_NULL);
				if (fcn) {
					RAnalRef *ref;
					RListIter *iter;
					RList *refs = r_anal_fcn_get_refs (core->anal, fcn);
					r_list_foreach (refs, iter, ref) {
						if (input[2] == 'j') {
							r_cons_printf ("{\"type\":\"%c\",\"from\":%"PFMT64d",\"to\":%"PFMT64d"}%s",
									ref->type, ref->at, ref->addr, iter->n? ",": "");
						} else {
							r_cons_printf ("%c 0x%08" PFMT64x " -> 0x%08" PFMT64x "\n",
									ref->type, ref->at, ref->addr);
						}
					}
					r_list_free (refs);
				} else {
					eprintf ("Cannot find function at 0x%08"PFMT64x"\n", addr);
				}
			}
			if (input[2] == 'j') {
				r_cons_printf ("]\n");
			}
#else
#warning TODO_ FCNOLD sdbize xrefs here
			eprintf ("TODO\n");
#endif
			break;
		case 'c': // "afxc" add code xref
		case 'd': // "afxd"
		case 's': // "afxs"
		case 'C': { // "afxC"
			char *p;
			ut64 a, b;
			char *mi = strdup (input);
			if (mi && mi[3] == ' ' && (p = strchr (mi + 4, ' '))) {
				*p = 0;
				a = r_num_math (core->num, mi + 3);
				b = r_num_math (core->num, p + 1);
				r_anal_xrefs_set (core->anal, input[2], a, b);
			} else {
				r_core_cmd_help (core, help_msg_afx);
			}
			free (mi);
			}
			break;
		case '-': // "afx-"
			{
			char *p;
			ut64 a, b;
			char *mi = strdup (input + 3);
			if (mi && *mi == ' ' && (p = strchr (mi + 1, ' '))) {
				*p = 0;
				a = r_num_math (core->num, mi);
				b = r_num_math (core->num, p + 1);
				r_anal_xrefs_deln (core->anal, -1, a, b);
			} else {
				eprintf ("Usage: afx- [src] [dst]\n");
			}
			free (mi);
			}
			break;
		default:
		case '?': // "afx?"
			r_core_cmd_help (core, help_msg_afx);
			break;
		}
		break;
	case 'F': // "afF"
		{
			int val = input[2] && r_num_math (core->num, input + 2);
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, R_ANAL_FCN_TYPE_NULL);
			if (fcn) {
				fcn->folded = input[2]? val: !fcn->folded;
			}
		}
		break;
	case '?': // "af?"
		r_core_cmd_help (core, help_msg_af);
		break;
	case 'r': // "afr" // analyze function recursively
	case ' ': // "af "
	case '\0': // "af"
		{
		char *uaddr = NULL, *name = NULL;
		int depth = r_config_get_i (core->config, "anal.depth");
		bool analyze_recursively = r_config_get_i (core->config, "anal.calls");
		RAnalFunction *fcn;
		ut64 addr = core->offset;
		if (input[1] == 'r') {
			input++;
			analyze_recursively = true;
		}

		// first undefine
		if (input[0] && input[1] == ' ') {
			name = strdup (input + 2);
			uaddr = strchr (name, ' ');
			if (uaddr) {
				*uaddr++ = 0;
				addr = r_num_math (core->num, uaddr);
			}
			// depth = 1; // or 1?
			// disable hasnext
		}

		//r_core_anal_undefine (core, core->offset);
		r_core_anal_fcn (core, addr, UT64_MAX, R_ANAL_REF_TYPE_NULL, depth);
		fcn = r_anal_get_fcn_in (core->anal, addr, 0);
		if (fcn && r_config_get_i (core->config, "anal.vars")) {
			fcn_callconv (core, fcn);
		}
		if (fcn) {
			/* ensure we use a proper name */
			setFunctionName (core, addr, fcn->name, false);
		}
		if (analyze_recursively) {
			fcn = r_anal_get_fcn_in (core->anal, addr, 0); /// XXX wrong in case of nopskip
			if (fcn) {
				RAnalRef *ref;
				RListIter *iter;
				RList *refs = r_anal_fcn_get_refs (core->anal, fcn);
				r_list_foreach (refs, iter, ref) {
					if (ref->addr == UT64_MAX) {
						//eprintf ("Warning: ignore 0x%08"PFMT64x" call 0x%08"PFMT64x"\n", ref->at, ref->addr);
						continue;
					}
					if (ref->type != 'c' && ref->type != 'C') {
						/* only follow code/call references */
						continue;
					}
					if (!r_io_is_valid_offset (core->io, ref->addr, !core->anal->opt.noncode)) {
						continue;
					}
					r_core_anal_fcn (core, ref->addr, fcn->addr, R_ANAL_REF_TYPE_CALL, depth);
					/* use recursivity here */
#if 1
					RAnalFunction *f = r_anal_get_fcn_at (core->anal, ref->addr, 0);
					if (f) {
						RListIter *iter;
						RAnalRef *ref;
						RList *refs1 = r_anal_fcn_get_refs (core->anal, f);
						r_list_foreach (refs1, iter, ref) {
							if (!r_io_is_valid_offset (core->io, ref->addr, !core->anal->opt.noncode)) {
								continue;
							}
							if (ref->type != 'c' && ref->type != 'C') {
								continue;
							}
							r_core_anal_fcn (core, ref->addr, f->addr, R_ANAL_REF_TYPE_CALL, depth);
							// recursively follow fcn->refs again and again
						}
						r_list_free (refs1);
					} else {
						f = r_anal_get_fcn_in (core->anal, fcn->addr, 0);
						if (f) {
							/* cut function */
							r_anal_fcn_resize (core->anal, f, addr - fcn->addr);
							r_core_anal_fcn (core, ref->addr, fcn->addr,
									R_ANAL_REF_TYPE_CALL, depth);
							f = r_anal_get_fcn_at (core->anal, fcn->addr, 0);
						}
						if (!f) {
							eprintf ("Cannot find function at 0x%08" PFMT64x "\n", fcn->addr);
						}
					}
#endif
				}
				r_list_free (refs);
			}
		}

		if (name) {
			if (*name && !setFunctionName (core, addr, name, true)) {
				eprintf ("Cannot find function '%s' at 0x%08" PFMT64x "\n", name, (ut64)addr);
			}
			free (name);
		}
		flag_every_function (core);
	}
	default:
		return false;
		break;
	}
	return true;
}

// size: 0: bits; -1: any; >0: exact size
static void __anal_reg_list(RCore *core, int type, int bits, char mode) {
	RReg *hack = core->dbg->reg;
	const char *use_color;
	int use_colors = r_config_get_i (core->config, "scr.color");
	if (use_colors) {
#undef ConsP
#define ConsP(x) (core->cons && core->cons->pal.x)? core->cons->pal.x
		use_color = ConsP (creg) : Color_BWHITE;
	} else {
		use_color = NULL;
	}
	if (bits < 0) {
		// TODO Change the `size` argument of r_debug_reg_list to use -1 for any and 0 for anal->bits
		bits = 0;
	} else if (!bits) {
		bits = core->anal->bits;
	}
	if (core->anal) {
		core->dbg->reg = core->anal->reg;
		if (core->anal->cur && core->anal->cur->arch) {
			/* workaround for thumb */
			if (!strcmp (core->anal->cur->arch, "arm") && bits == 16) {
				bits = 32;
			}
			/* workaround for 6502 */
			if (!strcmp (core->anal->cur->arch, "6502") && bits == 8) {
				r_debug_reg_list (core->dbg, R_REG_TYPE_GPR, 16, mode, use_color); // XXX detect which one is current usage
			}
			if (!strcmp (core->anal->cur->arch, "avr") && bits == 8) {
				r_debug_reg_list (core->dbg, R_REG_TYPE_GPR, 16, mode, use_color); // XXX detect which one is current usage
			}
		}
	}

	if (mode == '=') {
		int pcbits = 0;
		const char *pcname = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
		RRegItem *reg = r_reg_get (core->anal->reg, pcname, 0);
		if (bits != reg->size) {
			pcbits = reg->size;
		}
		if (pcbits) {
			r_debug_reg_list (core->dbg, R_REG_TYPE_GPR, pcbits, 2, use_color); // XXX detect which one is current usage
		}
	}
	r_debug_reg_list (core->dbg, type, bits, mode, use_color);
	core->dbg->reg = hack;
}

// XXX dup from drp :OOO
void cmd_anal_reg(RCore *core, const char *str) {
	int size = 0, i, type = R_REG_TYPE_GPR;
	int bits = (core->anal->bits & R_SYS_BITS_64)? 64: 32;
	int use_colors = r_config_get_i (core->config, "scr.color");
	struct r_reg_item_t *r;
	const char *use_color;
	const char *name;
	char *arg;

	if (use_colors) {
#define ConsP(x) (core->cons && core->cons->pal.x)? core->cons->pal.x
		use_color = ConsP (creg)
		: Color_BWHITE;
	} else {
		use_color = NULL;
	}
	switch (str[0]) {
	case 'l': // "arl"
	{
		RRegSet *rs = r_reg_regset_get (core->anal->reg, R_REG_TYPE_GPR);
		if (rs) {
			RRegItem *r;
			RListIter *iter;
			r_list_foreach (rs->regs, iter, r) {
				r_cons_println (r->name);
			}
		}
	} break;
	case '0': // "ar0"
		r_reg_arena_zero (core->anal->reg);
		break;
	case 'C': // "arC"
		if (core->anal->reg->reg_profile_cmt) {
			r_cons_println (core->anal->reg->reg_profile_cmt);
		}
		break;
	case 'w': // "arw"
		switch (str[1]) {
		case '?': {
			r_core_cmd_help (core, help_msg_arw);
			break;
		}
		case ' ':
			r_reg_arena_set_bytes (core->anal->reg, str + 1);
			break;
		default:
			r_core_cmd_help (core, help_msg_arw);
			break;
		}
		break;
	case 'a': // "ara"
		switch (str[1]) {
		case '?': // "ara?"
			r_core_cmd_help (core, help_msg_ara);
			break;
		case 's': // "aras"
			r_reg_arena_swap (core->anal->reg, false);
			break;
		case '+': // "ara+"
			r_reg_arena_push (core->anal->reg);
			break;
		case '-': // "ara-"
			r_reg_arena_pop (core->anal->reg);
			break;
		default: {
			int i, j;
			RRegArena *a;
			RListIter *iter;
			for (i = 0; i < R_REG_TYPE_LAST; i++) {
				RRegSet *rs = &core->anal->reg->regset[i];
				j = 0;
				r_list_foreach (rs->pool, iter, a) {
					r_cons_printf ("%s %p %d %d %s %d\n",
						(a == rs->arena)? "*": ".", a,
						i, j, r_reg_get_type (i), a->size);
					j++;
				}
			}
		} break;
		}
		break;
	case '?': // "ar?"
		if (str[1]) {
			ut64 off = r_reg_getv (core->anal->reg, str + 1);
			r_cons_printf ("0x%08" PFMT64x "\n", off);
		} else {
			r_core_cmd_help (core, help_msg_ar);
		}
		break;
	case 'r': // "arr"
		r_core_debug_rr (core, core->anal->reg);
		break;
	case 'S': { // "arS"
		int sz;
		ut8 *buf = r_reg_get_bytes (
			core->anal->reg, R_REG_TYPE_GPR, &sz);
		r_cons_printf ("%d\n", sz);
		free (buf);
		} break;
	case 'b': { // "arb" WORK IN PROGRESS // DEBUG COMMAND
		int len, type = R_REG_TYPE_GPR;
		arg = strchr (str, ' ');
		if (arg) {
			char *string = r_str_trim (strdup (arg + 1));
			if (string) {
				type = r_reg_type_by_name (string);
				if (type == -1 && string[0] != 'a') {
					type = R_REG_TYPE_GPR;
				}
				free (string);
			}
		}
		ut8 *buf = r_reg_get_bytes (core->dbg->reg, type, &len);
		if (buf) {
			//r_print_hexdump (core->print, 0LL, buf, len, 16, 16);
			r_print_hexdump (core->print, 0LL, buf, len, 32, 4, 1);
			free (buf);
		}
		} break;
	case 'c': // "arc"
		// TODO: set flag values with drc zf=1
		{
			RRegItem *r;
			const char *name = str + 1;
			while (*name == ' ') name++;
			if (*name && name[1]) {
				r = r_reg_cond_get (core->dbg->reg, name);
				if (r) {
					r_cons_println (r->name);
				} else {
					int id = r_reg_cond_from_string (name);
					RRegFlags *rf = r_reg_cond_retrieve (core->dbg->reg, NULL);
					if (rf) {
						int o = r_reg_cond_bits (core->dbg->reg, id, rf);
						core->num->value = o;
						// ORLY?
						r_cons_printf ("%d\n", o);
						free (rf);
					} else {
						eprintf ("unknown conditional or flag register\n");
					}
				}
			} else {
				RRegFlags *rf = r_reg_cond_retrieve (core->dbg->reg, NULL);
				if (rf) {
					r_cons_printf ("| s:%d z:%d c:%d o:%d p:%d\n",
						rf->s, rf->z, rf->c, rf->o, rf->p);
					if (*name == '=') {
						for (i = 0; i < R_REG_COND_LAST; i++) {
							r_cons_printf ("%s:%d ",
								r_reg_cond_to_string (i),
								r_reg_cond_bits (core->dbg->reg, i, rf));
						}
						r_cons_newline ();
					} else {
						for (i = 0; i < R_REG_COND_LAST; i++) {
							r_cons_printf ("%d %s\n",
								r_reg_cond_bits (core->dbg->reg, i, rf),
								r_reg_cond_to_string (i));
						}
					}
					free (rf);
				}
			}
		}
		break;
	case 's': // "ars"
		switch (str[1]) {
		case '-': // "ars-"
			r_reg_arena_pop (core->dbg->reg);
			// restore debug registers if in debugger mode
			r_debug_reg_sync (core->dbg, R_REG_TYPE_GPR, true);
			break;
		case '+': // "ars+"
			r_reg_arena_push (core->dbg->reg);
			break;
		case '?': { // "ars?"
			// TODO #7967 help refactor: dup from drp
			const char *help_msg[] = {
				"Usage:", "drs", " # Register states commands",
				"drs", "", "List register stack",
				"drs+", "", "Push register state",
				"drs-", "", "Pop register state",
				NULL };
			r_core_cmd_help (core, help_msg);
		} break;
		default:
			r_cons_printf ("%d\n", r_list_length (
						core->dbg->reg->regset[0].pool));
			break;
		}
		break;
	case 'p': // "arp"
		// XXX we have to break out .h for these cmd_xxx files.
		cmd_reg_profile (core, 'a', str);
		break;
	case 't': // "art"
		for (i = 0; (name = r_reg_get_type (i)); i++)
			r_cons_println (name);
		break;
	case 'n': // "arn"
		if (*(str + 1) == '\0') {
			eprintf ("Oops. try arn [PC|SP|BP|A0|A1|A2|A3|A4|R0|R1|ZF|SF|NF|OF]\n");
			break;
		}
		name = r_reg_get_name (core->dbg->reg, r_reg_get_name_idx (str + 2));
		if (name && *name) {
			r_cons_println (name);
		} else {
			eprintf ("Oops. try arn [PC|SP|BP|A0|A1|A2|A3|A4|R0|R1|ZF|SF|NF|OF]\n");
		}
		break;
	case 'd': // "ard"
		r_debug_reg_list (core->dbg, R_REG_TYPE_GPR, bits, 3, use_color); // XXX detect which one is current usage
		break;
	case 'o': // "aro"
		r_reg_arena_swap (core->dbg->reg, false);
		r_debug_reg_list (core->dbg, R_REG_TYPE_GPR, bits, 0, use_color); // XXX detect which one is current usage
		r_reg_arena_swap (core->dbg->reg, false);
		break;
	case '=': // "ar="
		{
			char *p = NULL;
			char *bits = NULL;
			if (str[1]) {
				p = strdup (str + 1);
				if (str[1] != ':') {
					// Bits were specified
					bits = strtok (p, ":");
					if (r_str_isnumber (bits)) {
						st64 sz = r_num_math (core->num, bits);
						if (sz > 0) {
							size = sz;
						}
					} else {
						r_core_cmd_help (core, help_msg_ar);
						break;
					}
				}
				int len = bits ? strlen (bits) : 0;
				if (str[len + 1] == ':') {
					// We have some regs
					char *regs = bits ? strtok (NULL, ":") : strtok ((char *)str + 1, ":");
					char *reg = strtok (regs, " ");
					RList *q_regs = r_list_new ();
					if (q_regs) {
						while (reg) {
							r_list_append (q_regs, reg);
							reg = strtok (NULL, " ");
						}
						core->dbg->q_regs = q_regs;
					}
				}
			}
			__anal_reg_list (core, type, size, 2);
			if (!r_list_empty (core->dbg->q_regs)) {
				r_list_free (core->dbg->q_regs);
			}
			core->dbg->q_regs = NULL;
			free (p);
		}
		break;
	case '-': // "ar-"
	case '*': // "ar*"
	case 'R': // "arR"
	case 'j': // "arj"
	case '\0': // "ar"
		__anal_reg_list (core, type, size, str[0]);
		break;
	case ' ': { // "ar "
		arg = strchr (str + 1, '=');
		if (arg) {
			char *ostr, *regname;
			*arg = 0;
			ostr = r_str_trim (strdup (str + 1));
			regname = r_str_trim_nc (ostr);
			r = r_reg_get (core->dbg->reg, regname, -1);
			if (!r) {
				int role = r_reg_get_name_idx (regname);
				if (role != -1) {
					const char *alias = r_reg_get_name (core->dbg->reg, role);
					r = r_reg_get (core->dbg->reg, alias, -1);
				}
			}
			if (r) {
				//eprintf ("%s 0x%08"PFMT64x" -> ", str,
				//	r_reg_get_value (core->dbg->reg, r));
				r_reg_set_value (core->dbg->reg, r,
						r_num_math (core->num, arg + 1));
				r_debug_reg_sync (core->dbg, R_REG_TYPE_ALL, true);
				//eprintf ("0x%08"PFMT64x"\n",
				//	r_reg_get_value (core->dbg->reg, r));
				r_core_cmdf (core, ".dr*%d", bits);
			} else {
				eprintf ("ar: Unknown register '%s'\n", regname);
			}
			free (ostr);
			return;
		}
		char name[32];
		int i = 1, j;
		while (str[i]) {
			if (str[i] == ',') {
				i++;
			} else {
				for (j = i; str[++j] && str[j] != ','; );
				if (j - i + 1 <= sizeof name) {
					r_str_ncpy (name, str + i, j - i + 1);
					if (IS_DIGIT (name[0])) { // e.g. ar 32
						__anal_reg_list (core, R_REG_TYPE_GPR, atoi (name), '\0');
					} else if (showreg (core, name) > 0) { // e.g. ar rax
					} else { // e.g. ar gpr ; ar all
						type = r_reg_type_by_name (name);
						// TODO differentiate ALL and illegal register types and print error message for the latter
						__anal_reg_list (core, type, -1, '\0');
					}
				}
				i = j;
			}
		}
	}
	}
}

static ut64 initializeEsil(RCore *core) {
	const char *name = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
	RAnalEsil *esil = core->anal->esil;
	int romem = r_config_get_i (core->config, "esil.romem");
	int stats = r_config_get_i (core->config, "esil.stats");
	int iotrap = r_config_get_i (core->config, "esil.iotrap");
	int exectrap = r_config_get_i (core->config, "esil.exectrap");
	int stacksize = r_config_get_i (core->config, "esil.stack.depth");
	int noNULL = r_config_get_i (core->config, "esil.noNULL");
	unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
	if (!(core->anal->esil = r_anal_esil_new (stacksize, iotrap, addrsize))) {
		return UT64_MAX;
	}
	ut64 addr;
	esil = core->anal->esil;
	r_anal_esil_setup (esil, core->anal, romem, stats, noNULL); // setup io
	esil->exectrap = exectrap;
	RList *entries = r_bin_get_entries (core->bin);
	RBinAddr *entry = NULL;
	RBinInfo *info = NULL;
	if (entries && !r_list_empty (entries)) {
		entry = (RBinAddr *)r_list_pop (entries);
		info = r_bin_get_info (core->bin);
		addr = info->has_va? entry->vaddr: entry->paddr;
		r_list_push (entries, entry);
	} else {
		addr = core->offset;
	}
	r_reg_setv (core->anal->reg, name, addr);
	// set memory read only
	return addr;
}

R_API int r_core_esil_step(RCore *core, ut64 until_addr, const char *until_expr, ut64 *prev_addr) {
#define return_tail(x) { tail_return_value = x; goto tail_return; }
	int tail_return_value = 0;
	int ret;
	ut8 code[32];
	RAnalOp op = {0};
	RAnalEsil *esil = core->anal->esil;
	const char *name = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
	if (!esil) {
// TODO		inititalizeEsil (core);

		int stacksize = r_config_get_i (core->config, "esil.stack.depth");
		int iotrap = r_config_get_i (core->config, "esil.iotrap");
		int romem = r_config_get_i (core->config, "esil.romem");
		int stats = r_config_get_i (core->config, "esil.stats");
		int noNULL = r_config_get_i (core->config, "esil.noNULL");
		int verbose = r_config_get_i (core->config, "esil.verbose");
		unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
		if (!(esil = r_anal_esil_new (stacksize, iotrap, addrsize))) {
			return 0;
		}
		r_anal_esil_setup (esil, core->anal, romem, stats, noNULL); // setup io
		core->anal->esil = esil;
		esil->verbose = verbose;
		{
			const char *s = r_config_get (core->config, "cmd.esil.intr");
			if (s) {
				char *my = strdup (s);
				if (my) {
					r_config_set (core->config, "cmd.esil.intr", my);
					free (my);
				}
			}
		}
	}
	esil->cmd = r_core_esil_cmd;
	ut64 addr = r_reg_getv (core->anal->reg, name);
	r_cons_break_push (NULL, NULL);
repeat:
	if (r_cons_is_breaked ()) {
		eprintf ("[+] ESIL emulation interrupted at 0x%08" PFMT64x "\n", addr);
		return_tail (0);
	}
	if (!esil) {
		addr = initializeEsil (core);
		esil = core->anal->esil;
		if (!esil) {
			return_tail (0);
		}
	} else {
		esil->trap = 0;
		addr = r_reg_getv (core->anal->reg, name);
		//eprintf ("PC=0x%"PFMT64x"\n", (ut64)addr);
	}
	if (prev_addr) {
		*prev_addr = addr;
	}
	if (esil->exectrap) {
		if (!r_io_is_valid_offset (core->io, addr, R_IO_EXEC)) {
			esil->trap = R_ANAL_TRAP_EXEC_ERR;
			esil->trap_code = addr;
			eprintf ("[ESIL] Trap, trying to execute on non-executable memory\n");
			return_tail (1);
		}
	}
	r_asm_set_pc (core->assembler, addr);
	// run esil pin command here
	const char *pincmd = r_anal_pin_call (core->anal, addr);
	if (pincmd) {
		r_core_cmd0 (core, pincmd);
		ut64 pc = r_debug_reg_get (core->dbg, "PC");
		if (addr != pc) {
			return_tail (1);
		}
	}
	(void)r_io_read_at (core->io, addr, code, sizeof (code));
	// TODO: sometimes this is dupe
	ret = r_anal_op (core->anal, &op, addr, code, sizeof (code), R_ANAL_OP_MASK_ALL);
// if type is JMP then we execute the next N instructions
	// update the esil pointer because RAnal.op() can change it
	esil = core->anal->esil;
	if (op.size < 1 || ret < 0) {
		if (esil->cmd && esil->cmd_todo) {
			esil->cmd (esil, esil->cmd_todo, addr, 0);
		}
		op.size = 1; // avoid inverted stepping
	}
	{
		/* apply hint */
		RAnalHint *hint = r_anal_hint_get (core->anal, addr);
		r_anal_op_hint (&op, hint);
		r_anal_hint_free (hint);
	}
	r_reg_setv (core->anal->reg, name, addr + op.size);
	if (ret) {
		r_anal_esil_set_pc (esil, addr);
		if (core->dbg->trace->enabled) {
			RReg *reg = core->dbg->reg;
			core->dbg->reg = core->anal->reg;
			r_debug_trace_pc (core->dbg, addr);
			core->dbg->reg = reg;
		} else {
			r_anal_esil_parse (esil, R_STRBUF_SAFEGET (&op.esil));

			if (core->anal->cur && core->anal->cur->esil_post_loop) {
				core->anal->cur->esil_post_loop (esil, &op);
			}
			r_anal_esil_stack_free (esil);
		}
		// only support 1 slot for now
		if (op.delay) {
			ut8 code2[32];
			ut64 naddr = addr + op.size;
			RAnalOp op2 = {0};
			// emulate only 1 instruction
			r_anal_esil_set_pc (esil, naddr);
			(void)r_io_read_at (core->io, naddr, code2, sizeof (code2));
			// TODO: sometimes this is dupe
			ret = r_anal_op (core->anal, &op2, naddr, code2, sizeof (code2), R_ANAL_OP_MASK_ALL);
			switch (op2.type) {
			case R_ANAL_OP_TYPE_CJMP:
			case R_ANAL_OP_TYPE_JMP:
			case R_ANAL_OP_TYPE_CRET:
			case R_ANAL_OP_TYPE_RET:
				// branches are illegal in a delay slot
				esil->trap = R_ANAL_TRAP_EXEC_ERR;
				esil->trap_code = addr;
				eprintf ("[ESIL] Trap, trying to execute a branch in a delay slot\n");
				return_tail (1);
				break;
			}
			r_anal_esil_parse (esil, R_STRBUF_SAFEGET (&op2.esil));
			r_anal_op_fini (&op2);
		}
		tail_return_value = 1;
	}

	st64 follow = (st64)r_config_get_i (core->config, "dbg.follow");
	ut64 pc = r_debug_reg_get (core->dbg, "PC");
	if (follow > 0) {
		if ((pc < core->offset) || (pc > (core->offset + follow))) {
			r_core_cmd0 (core, "sr PC");
		}
	}

	// check addr
	if (until_addr != UT64_MAX) {
		if (r_reg_getv (core->anal->reg, name) == until_addr) {
			return_tail (0);
		}
		goto repeat;
	}
	// check esil
	if (esil && esil->trap) {
		if (core->anal->esil->verbose) {
			eprintf ("TRAP\n");
		}
		return_tail (0);
	}
	if (until_expr) {
		if (r_anal_esil_condition (core->anal->esil, until_expr)) {
			if (core->anal->esil->verbose) {
				eprintf ("ESIL BREAK!\n");
			}
			return_tail (0);
		}
		goto repeat;
	}
tail_return:
	r_anal_op_fini (&op);
	r_cons_break_pop ();
	return tail_return_value;
}

R_API int r_core_esil_step_back(RCore *core) {
	RAnalEsil *esil = core->anal->esil;
	RListIter *tail;
	const char *name = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
	ut64 prev = 0;
	ut64 end = r_reg_getv (core->anal->reg, name);

	if (!esil || !(tail = r_list_tail (esil->sessions))) {
		return 0;
	}
	RAnalEsilSession *before = (RAnalEsilSession *) tail->data;
	if (!before) {
		eprintf ("Cannot find any previous state here\n");
		return 0;
	}
	eprintf ("NOTE: step back in esil is setting an initial state and stepping into pc is the same.\n");
	eprintf ("NOTE: this is extremely wrong and poorly efficient. so don't use this feature unless\n");
	eprintf ("NOTE: you are going to fix it by making it consistent with dts, which is also broken as hell\n");
	eprintf ("Execute until 0x%08"PFMT64x"\n", end);
	r_anal_esil_session_set (esil, before);
	r_core_esil_step (core, end, NULL, &prev);
	eprintf ("Before 0x%08"PFMT64x"\n", prev);
	r_anal_esil_session_set (esil, before);
	r_core_esil_step (core, prev, NULL, NULL);
	return 1;
}

static void cmd_address_info(RCore *core, const char *addrstr, int fmt) {
	ut64 addr, type;
	if (!addrstr || !*addrstr) {
		addr = core->offset;
	} else {
		addr = r_num_math (core->num, addrstr);
	}
	type = r_core_anal_address (core, addr);
	int isp = 0;
	switch (fmt) {
	case 'j':
#define COMMA isp++? ",": ""
		r_cons_printf ("{");
		if (type & R_ANAL_ADDR_TYPE_PROGRAM)
			r_cons_printf ("%s\"program\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_LIBRARY)
			r_cons_printf ("%s\"library\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_EXEC)
			r_cons_printf ("%s\"exec\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_READ)
			r_cons_printf ("%s\"read\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_WRITE)
			r_cons_printf ("%s\"write\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_FLAG)
			r_cons_printf ("%s\"flag\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_FUNC)
			r_cons_printf ("%s\"func\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_STACK)
			r_cons_printf ("%s\"stack\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_HEAP)
			r_cons_printf ("%s\"heap\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_REG)
			r_cons_printf ("%s\"reg\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_ASCII)
			r_cons_printf ("%s\"ascii\":true", COMMA);
		if (type & R_ANAL_ADDR_TYPE_SEQUENCE)
			r_cons_printf ("%s\"sequence\":true", COMMA);
		r_cons_print ("}");
		break;
	default:
		if (type & R_ANAL_ADDR_TYPE_PROGRAM)
			r_cons_printf ("program\n");
		if (type & R_ANAL_ADDR_TYPE_LIBRARY)
			r_cons_printf ("library\n");
		if (type & R_ANAL_ADDR_TYPE_EXEC)
			r_cons_printf ("exec\n");
		if (type & R_ANAL_ADDR_TYPE_READ)
			r_cons_printf ("read\n");
		if (type & R_ANAL_ADDR_TYPE_WRITE)
			r_cons_printf ("write\n");
		if (type & R_ANAL_ADDR_TYPE_FLAG)
			r_cons_printf ("flag\n");
		if (type & R_ANAL_ADDR_TYPE_FUNC)
			r_cons_printf ("func\n");
		if (type & R_ANAL_ADDR_TYPE_STACK)
			r_cons_printf ("stack\n");
		if (type & R_ANAL_ADDR_TYPE_HEAP)
			r_cons_printf ("heap\n");
		if (type & R_ANAL_ADDR_TYPE_REG)
			r_cons_printf ("reg\n");
		if (type & R_ANAL_ADDR_TYPE_ASCII)
			r_cons_printf ("ascii\n");
		if (type & R_ANAL_ADDR_TYPE_SEQUENCE)
			r_cons_printf ("sequence\n");
	}
}

static void cmd_anal_info(RCore *core, const char *input) {
	switch (input[0]) {
	case '?':
		eprintf ("Usage: ai @ rsp\n");
		break;
	case ' ':
		cmd_address_info (core, input, 0);
		break;
	case 'j': // "aij"
		cmd_address_info (core, input + 1, 'j');
		break;
	default:
		cmd_address_info (core, NULL, 0);
		break;
	}
}

static void initialize_stack (RCore *core, ut64 addr, ut64 size) {
	const char *mode = r_config_get (core->config, "esil.fillstack");
	if (mode && *mode && *mode != '0') {
		const int bs = 4096 * 32;
		ut64 i;
		for (i = 0; i < size; i += bs) {
			int left = R_MIN (bs, size - i);
		//	r_core_cmdf (core, "wx 10203040 @ 0x%llx", addr);
			switch (*mode) {
			case 'd': // "debrujn"
				r_core_cmdf (core, "wopD %"PFMT64d" @ 0x%"PFMT64x, left, addr + i);
				break;
			case 's': // "seq"
				r_core_cmdf (core, "woe 1 0xff 1 4 @ 0x%"PFMT64x"!0x%"PFMT64x, addr + i, left);
				break;
			case 'r': // "random"
				r_core_cmdf (core, "woR %"PFMT64d" @ 0x%"PFMT64x"!0x%"PFMT64x, left, addr + i, left);
				break;
			case 'z': // "zero"
			case '0':
				r_core_cmdf (core, "wow 00 @ 0x%"PFMT64x"!0x%"PFMT64x, addr + i, left);
				break;
			}
		}
		// eprintf ("[*] Initializing ESIL stack with pattern\n");
		// r_core_cmdf (core, "woe 0 10 4 @ 0x%"PFMT64x, size, addr);
	}
}

static void cmd_esil_mem(RCore *core, const char *input) {
	RAnalEsil *esil = core->anal->esil;
	RIOMap *stack_map;
	ut64 curoff = core->offset;
	const char *patt = "";
	ut64 addr = 0x100000;
	ut32 size = 0xf0000;
	char name[128];
	RFlagItem *fi;
	const char *sp, *pc;
	char uri[32];
	char nomalloc[256];
	char *p;
	if (!esil) {
		int stacksize = r_config_get_i (core->config, "esil.stack.depth");
		int iotrap = r_config_get_i (core->config, "esil.iotrap");
		int romem = r_config_get_i (core->config, "esil.romem");
		int stats = r_config_get_i (core->config, "esil.stats");
		int noNULL = r_config_get_i (core->config, "esil.noNULL");
		int verbose = r_config_get_i (core->config, "esil.verbose");
		unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
		if (!(esil = r_anal_esil_new (stacksize, iotrap, addrsize))) {
			return;
		}
		r_anal_esil_setup (esil, core->anal, romem, stats, noNULL); // setup io
		core->anal->esil = esil;
		esil->verbose = verbose;
		{
			const char *s = r_config_get (core->config, "cmd.esil.intr");
			if (s) {
				char *my = strdup (s);
				if (my) {
					r_config_set (core->config, "cmd.esil.intr", my);
					free (my);
				}
			}
		}
	}
	if (*input == '?') {
		eprintf ("Usage: aeim [addr] [size] [name] - initialize ESIL VM stack\n");
		eprintf ("Default: 0x100000 0xf0000\n");
		eprintf ("See ae? for more help\n");
		return;
	}

	if (input[0] == 'p') {
		fi = r_flag_get (core->flags, "aeim.stack");
		if (fi) {
			addr = fi->offset;
			size = fi->size;
		} else {
			cmd_esil_mem (core, "");
		}
		if (esil) {
			esil->stack_addr = addr;
			esil->stack_size = size;
		}
		initialize_stack (core, addr, size);
		return;
	}

	if (!*input) {
		RFlagItem *fi = r_flag_get (core->flags, "aeim.fd");
		if (fi) {
			// Close the fd associated with the aeim stack
			(void)r_io_fd_close (core->io, fi->offset);
		}
	}
	addr = r_config_get_i (core->config, "esil.stack.addr");
	size = r_config_get_i (core->config, "esil.stack.size");
	patt = r_config_get (core->config, "esil.stack.pattern");

	p = strncpy (nomalloc, input, 255);
	if ((p = strchr (p, ' '))) {
		while (*p == ' ') p++;
		addr = r_num_math (core->num, p);
		if ((p = strchr (p, ' '))) {
			while (*p == ' ') p++;
			size = (ut32)r_num_math (core->num, p);
			if (size < 1) {
				size = 0xf0000;
			}
			if ((p = strchr (p, ' '))) {
				while (*p == ' ') p++;
				snprintf (name, sizeof (name), "mem.%s", p);
			} else {
				snprintf (name, sizeof (name), "mem.0x%" PFMT64x "_0x%x", addr, size);
			}
		} else {
			snprintf (name, sizeof (name), "mem.0x%" PFMT64x "_0x%x", addr, size);
		}
	} else {
		snprintf (name, sizeof (name), "mem.0x%" PFMT64x "_0x%x", addr, size);
	}
	if (*input == '-') {
		if (esil->stack_fd > 2) {	//0, 1, 2 are reserved for stdio/stderr
			r_io_fd_close (core->io, esil->stack_fd);
			// no need to kill the maps, r_io_map_cleanup does that for us in the close
			esil->stack_fd = 0;
		} else {
			eprintf ("Cannot deinitialize %s\n", name);
		}
		r_flag_unset_name (core->flags, name);
		// eprintf ("Deinitialized %s\n", name);
		return;
	}

	snprintf (uri, sizeof (uri), "malloc://%d", (int)size);
	esil->stack_fd = r_io_fd_open (core->io, uri, R_IO_RW, 0);
	if (!(stack_map = r_io_map_add (core->io, esil->stack_fd,
			R_IO_RW, 0LL, addr, size, true))) {
		r_io_fd_close (core->io, esil->stack_fd);
		eprintf ("Cannot create map for tha stack, fd %d got closed again\n", esil->stack_fd);
		esil->stack_fd = 0;
		return;
	}
	r_io_map_set_name (stack_map, name);
	// r_flag_set (core->flags, name, addr, size);	//why is this here?
	r_flag_set (core->flags, "aeim.stack", addr, size);
	r_flag_set (core->flags, "aeim.fd", esil->stack_fd, 1);
	r_config_set_i (core->config, "io.va", true);
	if (patt && *patt) {
		switch (*patt) {
		case '0':
			// do nothing
			break;
		case 'd':
			r_core_cmdf (core, "wopD %d @ 0x%"PFMT64x, size, addr);
			break;
		case 'i':
			r_core_cmdf (core, "woe 0 255 1 @ 0x%"PFMT64x"!%d",addr, size);
			break;
		case 'w':
			r_core_cmdf (core, "woe 0 0xffff 1 4 @ 0x%"PFMT64x"!%d",addr, size);
			break;
		}
	}
	// SP
	sp = r_reg_get_name (core->dbg->reg, R_REG_NAME_SP);
	r_debug_reg_set (core->dbg, sp, addr + (size / 2));
	// BP
	sp = r_reg_get_name (core->dbg->reg, R_REG_NAME_BP);
	r_debug_reg_set (core->dbg, sp, addr + (size / 2));
	// PC
	pc = r_reg_get_name (core->dbg->reg, R_REG_NAME_PC);
	r_debug_reg_set (core->dbg, pc, curoff);
	r_core_cmd0 (core, ".ar*");
#if 0
	if (!r_io_section_get_name (core->io, ESIL_STACK_NAME)) {
		r_core_cmdf (core, "om %d 0x%"PFMT64x, cf->fd, addr);
		r_core_cmdf (core, "S 0x%"PFMT64x" 0x%"PFMT64x" %d %d "
			ESIL_STACK_NAME, addr, addr, size, size);
	}
#endif
	if (esil) {
		esil->stack_addr = addr;
		esil->stack_size = size;
	}
	initialize_stack (core, addr, size);
	r_core_seek (core, curoff, 0);
}

#if 0
static ut64 opc = UT64_MAX;
static ut8 *regstate = NULL;

static void esil_init (RCore *core) {
	const char *pc = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
	int noNULL = r_config_get_i (core->config, "esil.noNULL");
	opc = r_reg_getv (core->anal->reg, pc);
	if (!opc || opc==UT64_MAX) {
		opc = core->offset;
	}
	if (!core->anal->esil) {
		int iotrap = r_config_get_i (core->config, "esil.iotrap");
		ut64 stackSize = r_config_get_i (core->config, "esil.stack.size");
		unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
		if (!(core->anal->esil = r_anal_esil_new (stackSize, iotrap, addrsize))) {
			R_FREE (regstate);
			return;
		}
		r_anal_esil_setup (core->anal->esil, core->anal, 0, 0, noNULL);
	}
	free (regstate);
	regstate = r_reg_arena_peek (core->anal->reg);
}

static void esil_fini(RCore *core) {
	const char *pc = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
	r_reg_arena_poke (core->anal->reg, regstate);
	r_reg_setv (core->anal->reg, pc, opc);
	R_FREE (regstate);
}
#endif

typedef struct {
	RList *regs;
	RList *regread;
	RList *regwrite;
	RList *inputregs;
} AeaStats;

static void aea_stats_init (AeaStats *stats) {
	stats->regs = r_list_newf (free);
	stats->regread = r_list_newf (free);
	stats->regwrite = r_list_newf (free);
	stats->inputregs = r_list_newf (free);
}

static void aea_stats_fini (AeaStats *stats) {
	R_FREE (stats->regs);
	R_FREE (stats->regread);
	R_FREE (stats->regwrite);
	R_FREE (stats->inputregs);
}

static bool contains(RList *list, const char *name) {
	RListIter *iter;
	const char *n;
	r_list_foreach (list, iter, n) {
		if (!strcmp (name, n))
			return true;
	}
	return false;
}

static char *oldregread = NULL;
static RList *mymemxsr = NULL;
static RList *mymemxsw = NULL;

#define R_NEW_DUP(x) memcpy((void*)malloc(sizeof(x)), &(x), sizeof(x))
typedef struct {
	ut64 addr;
	int size;
} AeaMemItem;

static int mymemwrite(RAnalEsil *esil, ut64 addr, const ut8 *buf, int len) {
	RListIter *iter;
	AeaMemItem *n;
	r_list_foreach (mymemxsw, iter, n) {
		if (addr == n->addr) {
			return len;
		}
	}
	if (!r_io_is_valid_offset (esil->anal->iob.io, addr, 0)) {
		return false;
	}
	n = R_NEW (AeaMemItem);
	if (n) {
		n->addr = addr;
		n->size = len;
		r_list_push (mymemxsw, n);
	}
	return len;
}

static int mymemread(RAnalEsil *esil, ut64 addr, ut8 *buf, int len) {
	RListIter *iter;
	AeaMemItem *n;
	r_list_foreach (mymemxsr, iter, n) {
		if (addr == n->addr) {
			return len;
		}
	}
	if (!r_io_is_valid_offset (esil->anal->iob.io, addr, 0)) {
		return false;
	}
	n = R_NEW (AeaMemItem);
	if (n) {
		n->addr = addr;
		n->size = len;
		r_list_push (mymemxsr, n);
	}
	return len;
}

static int myregwrite(RAnalEsil *esil, const char *name, ut64 *val) {
	AeaStats *stats = esil->user;
	if (oldregread && !strcmp (name, oldregread)) {
		r_list_pop (stats->regread);
		R_FREE (oldregread)
	}
	if (!IS_DIGIT (*name)) {
		if (!contains (stats->regs, name)) {
			r_list_push (stats->regs, strdup (name));
		}
		if (!contains (stats->regwrite, name)) {
			r_list_push (stats->regwrite, strdup (name));
		}
	}
	return 0;
}

static int myregread(RAnalEsil *esil, const char *name, ut64 *val, int *len) {
	AeaStats *stats = esil->user;
	if (!IS_DIGIT (*name)) {
		if (!contains (stats->inputregs, name)) {
			if (!contains (stats->regwrite, name)) {
				r_list_push (stats->inputregs, strdup (name));
			}
		}
		if (!contains (stats->regs, name)) {
			r_list_push (stats->regs, strdup (name));
		}
		if (!contains (stats->regread, name)) {
			r_list_push (stats->regread, strdup (name));
		}
	}
	return 0;
}

static void showregs (RList *list) {
	if (!r_list_empty (list)) {
		char *reg;
		RListIter *iter;
		r_list_foreach (list, iter, reg) {
			r_cons_print (reg);
			if (iter->n) {
				r_cons_printf (" ");
			}
		}
	}
	r_cons_newline();
}

static void showregs_json (RList *list) {
	r_cons_printf ("[");
	if (!r_list_empty (list)) {
		char *reg;
		RListIter *iter;
		r_list_foreach (list, iter, reg) {
			r_cons_printf ("\"%s\"", reg);
			if (iter->n) {
				r_cons_printf (",");
			}
		}
	}
	r_cons_printf ("]");
}

static bool cmd_aea(RCore* core, int mode, ut64 addr, int length) {
	RAnalEsil *esil;
	int ptr, ops, ops_end = 0, len, buf_sz, maxopsize;
	ut64 addr_end;
	AeaStats stats;
	const char *esilstr;
	RAnalOp aop = R_EMPTY;
	ut8 *buf;
	RList* regnow;
	if (!core) {
		return false;
	}
	maxopsize = r_anal_archinfo (core->anal, R_ANAL_ARCHINFO_MAX_OP_SIZE);
	if (maxopsize < 1) {
		maxopsize = 16;
	}
	if (mode & 1) {
		// number of bytes / length
		buf_sz = length;
	} else {
		// number of instructions / opcodes
		ops_end = length;
		if (ops_end < 1) {
			ops_end = 1;
		}
		buf_sz = ops_end * maxopsize;
	}
	if (buf_sz < 1) {
		buf_sz = maxopsize;
	}
	addr_end = addr + buf_sz;
	buf = malloc (buf_sz);
	if (!buf) {
		return false;
	}
	(void)r_io_read_at (core->io, addr, (ut8 *)buf, buf_sz);
	aea_stats_init (&stats);

	//esil_init (core);
	//esil = core->anal->esil;
	r_reg_arena_push (core->anal->reg);
	int stacksize = r_config_get_i (core->config, "esil.stack.depth");
	bool iotrap = r_config_get_i (core->config, "esil.iotrap");
	int romem = r_config_get_i (core->config, "esil.romem");
	int stats1 = r_config_get_i (core->config, "esil.stats");
	int noNULL = r_config_get_i (core->config, "esil.noNULL");
	unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");
	esil = r_anal_esil_new (stacksize, iotrap, addrsize);
	r_anal_esil_setup (esil, core->anal, romem, stats1, noNULL); // setup io
#	define hasNext(x) (x&1) ? (addr<addr_end) : (ops<ops_end)

	mymemxsr = r_list_new ();
	mymemxsw = r_list_new ();
	esil->user = &stats;
	esil->cb.hook_reg_write = myregwrite;
	esil->cb.hook_reg_read = myregread;
	esil->cb.hook_mem_write = mymemwrite;
	esil->cb.hook_mem_read = mymemread;
	esil->nowrite = true;
	for (ops = ptr = 0; ptr < buf_sz && hasNext (mode); ops++, ptr += len) {
		len = r_anal_op (core->anal, &aop, addr + ptr, buf + ptr, buf_sz - ptr, R_ANAL_OP_MASK_ALL);
		esilstr = R_STRBUF_SAFEGET (&aop.esil);
		if (len < 1) {
			eprintf ("Invalid 0x%08"PFMT64x" instruction %02x %02x\n",
				addr + ptr, buf[ptr], buf[ptr + 1]);
			break;
		}
		r_anal_esil_parse (esil, esilstr);
		r_anal_esil_stack_free (esil);
	}
	esil->nowrite = false;
	esil->cb.hook_reg_write = NULL;
	esil->cb.hook_reg_read = NULL;
	//esil_fini (core);
	r_anal_esil_free (esil);
	r_reg_arena_pop (core->anal->reg);
	regnow = r_list_newf (free);
	{
		RListIter *iter;
		char *reg;
		r_list_foreach (stats.regs, iter, reg) {
			if (!contains (stats.regwrite, reg)) {
				r_list_push (regnow, strdup (reg));
			}
		}
	}
	if ((mode >> 5) & 1) {
		RListIter *iter;
		AeaMemItem *n;
		int c = 0;
		r_cons_printf ("f-mem.*\n");
		r_list_foreach (mymemxsr, iter, n) {
			r_cons_printf ("f mem.read.%d 0x%08x @ 0x%08"PFMT64x"\n", c++, n->size, n->addr);
		}
		c = 0;
		r_list_foreach (mymemxsw, iter, n) {
			r_cons_printf ("f mem.write.%d 0x%08x @ 0x%08"PFMT64x"\n", c++, n->size, n->addr);
		}
	}

	/* show registers used */
	if ((mode >> 1) & 1) {
		showregs (stats.regread);
	} else if ((mode >> 2) & 1) {
		showregs (stats.regwrite);
	} else if ((mode >> 3) & 1) {
		showregs (regnow);
	} else if ((mode >> 4) & 1) {
		r_cons_printf ("{\"A\":");
		showregs_json (stats.regs);
		r_cons_printf (",\"I\":");
		showregs_json (stats.inputregs);
		r_cons_printf (",\"R\":");
		showregs_json (stats.regread);
		r_cons_printf (",\"W\":");
		showregs_json (stats.regwrite);
		r_cons_printf (",\"N\":");
		showregs_json (regnow);
		r_cons_printf ("}");
		r_cons_newline();
	} else if ((mode >> 5) & 1) {
		// nothing
	} else {
		r_cons_printf (" I: ");
		showregs (stats.inputregs);
		r_cons_printf (" A: ");
		showregs (stats.regs);
		r_cons_printf (" R: ");
		showregs (stats.regread);
		r_cons_printf (" W: ");
		showregs (stats.regwrite);
		r_cons_printf ("NW: ");
		if (r_list_length (regnow)) {
			showregs (regnow);
		} else {
			r_cons_newline();
		}
		RListIter *iter;
		ut64 *n;
		if (!r_list_empty (mymemxsr)) {
			r_cons_printf ("@R:");
			r_list_foreach (mymemxsr, iter, n) {
				r_cons_printf (" 0x%08"PFMT64x, *n);
			}
			r_cons_newline ();
		}
		if (!r_list_empty (mymemxsw)) {
			r_cons_printf ("@W:");
			r_list_foreach (mymemxsw, iter, n) {
				r_cons_printf (" 0x%08"PFMT64x, *n);
			}
			r_cons_newline ();
		}
	}
	r_list_free (mymemxsr);
	r_list_free (mymemxsw);
	mymemxsr = NULL;
	mymemxsw = NULL;
	aea_stats_fini (&stats);
	free (buf);
	R_FREE (regnow);
	return true;
}

static void cmd_aespc(RCore *core, ut64 addr, int off) {
	RAnalEsil *esil = core->anal->esil;
	int i, j = 0;
	int instr_size = 0;
	ut8 *buf;
	RAnalOp aop = {0};
	int ret , bsize = R_MAX (64, core->blocksize);
	const int mininstrsz = r_anal_archinfo (core->anal, R_ANAL_ARCHINFO_MIN_OP_SIZE);
	const int minopcode = R_MAX (1, mininstrsz);
	const char *pc = r_reg_get_name (core->dbg->reg, R_REG_NAME_PC);
	RRegItem *r = r_reg_get (core->dbg->reg, pc, -1);
	int stacksize = r_config_get_i (core->config, "esil.stack.depth");
	int iotrap = r_config_get_i (core->config, "esil.iotrap");
	unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");

	if (!esil) {
		if (!(esil = r_anal_esil_new (stacksize, iotrap, addrsize))) {
			return;
		}
	}
	buf = malloc (bsize);
	if (!buf) {
		eprintf ("Cannot allocate %d byte(s)\n", bsize);
		free (buf);
		return;
	}
	if (addr == -1) {
		addr = r_debug_reg_get (core->dbg, pc);
	}
	ut64 curpc = addr;
	ut64 oldoff = core->offset;
	for (i = 0, j = 0; j < off ; i++, j++) {
		if (r_cons_is_breaked ()) {
			break;
		}
		if (i >= (bsize - 32)) {
			i = 0;
		}
		if (!i) {
			r_core_read_at (core, addr, buf, bsize);
		}
		ret = r_anal_op (core->anal, &aop, addr, buf + i, bsize - i, R_ANAL_OP_MASK_ALL);
		instr_size += ret;
		int inc = (core->search->align > 0)? core->search->align - 1: ret - 1;
		if (inc < 0) {
			inc = minopcode;
		}
		i += inc;
		addr += inc;
		r_anal_op_fini (&aop);
	}
	r_reg_set_value (core->dbg->reg, r, curpc);
	r_core_esil_step (core, curpc + instr_size, NULL, NULL);
	r_core_seek (core, oldoff, 1);
}

static void cmd_anal_esil(RCore *core, const char *input) {
	RAnalEsil *esil = core->anal->esil;
	ut64 addr = core->offset;
	ut64 adr ;
	char *n, *n1;
	int off;
	int stacksize = r_config_get_i (core->config, "esil.stack.depth");
	int iotrap = r_config_get_i (core->config, "esil.iotrap");
	int romem = r_config_get_i (core->config, "esil.romem");
	int stats = r_config_get_i (core->config, "esil.stats");
	int noNULL = r_config_get_i (core->config, "esil.noNULL");
	ut64 until_addr = UT64_MAX;
	unsigned int addrsize = r_config_get_i (core->config, "esil.addr.size");

	const char *until_expr = NULL;
	RAnalOp *op;

	switch (input[0]) {
	case 'p': // "aep"
		switch (input[1]) {
		case 'c':
			if (input[2] == ' ') {
				// seek to this address
				r_core_cmdf (core, "ar PC=%s", input + 3);
				r_core_cmd0 (core, ".ar*");
			} else {
				eprintf ("Missing argument\n");
			}
			break;
		case 0:
			r_anal_pin_list (core->anal);
			break;
		case '-':
			if (input[2])
				addr = r_num_math (core->num, input + 2);
			r_anal_pin_unset (core->anal, addr);
			break;
		case ' ':
			r_anal_pin (core->anal, addr, input + 2);
			break;
		default:
			r_core_cmd_help (core, help_msg_aep);
			break;
		}
		break;
	case 'r': // "aer"
		// 'aer' is an alias for 'ar'
		cmd_anal_reg (core, input + 1);
		break;
	case '*':
		// XXX: this is wip, not working atm
		if (core->anal->esil) {
			r_cons_printf ("trap: %d\n", core->anal->esil->trap);
			r_cons_printf ("trap-code: %d\n", core->anal->esil->trap_code);
		} else {
			eprintf ("esil vm not initialized. run `aei`\n");
		}
		break;
	case ' ':
		//r_anal_esil_eval (core->anal, input+1);
		if (!esil) {
			if (!(core->anal->esil = esil = r_anal_esil_new (stacksize, iotrap, addrsize)))
				return;
		}
		r_anal_esil_setup (esil, core->anal, romem, stats, noNULL); // setup io
		r_anal_esil_set_pc (esil, core->offset);
		r_anal_esil_parse (esil, input + 1);
		r_anal_esil_dumpstack (esil);
		r_anal_esil_stack_free (esil);
		break;
	case 's': // "aes"
		// "aes" "aeso" "aesu" "aesue"
		// aes -> single step
		// aesb -> single step back
		// aeso -> single step over
		// aesu -> until address
		// aesue -> until esil expression
		switch (input[1]) {
		case '?':
			eprintf ("See: ae?~aes\n");
			break;
		case 'l': // "aesl"
		{
			ut64 pc = r_debug_reg_get (core->dbg, "PC");
			RAnalOp *op = r_core_anal_op (core, pc);
// TODO: honor hint
			if (!op) {
				break;
			}
			r_core_esil_step (core, UT64_MAX, NULL, NULL);
			r_debug_reg_set (core->dbg, "PC", pc + op->size);
			r_anal_esil_set_pc (esil, pc + op->size);
			r_core_cmd0 (core, ".ar*");
		} break;
		case 'b': // "aesb"
			if (!r_core_esil_step_back (core)) {
				eprintf ("cannnot step back\n");
			}
			r_core_cmd0 (core, ".ar*");
			break;
		case 'u': // "aesu"
			if (input[2] == 'e') {
				until_expr = input + 3;
			} else {
				until_addr = r_num_math (core->num, input + 2);
			}
			r_core_esil_step (core, until_addr, until_expr, NULL);
			r_core_cmd0 (core, ".ar*");
			break;
		case 'o': // "aeso"
			// step over
			op = r_core_anal_op (core, r_reg_getv (core->anal->reg,
				r_reg_get_name (core->anal->reg, R_REG_NAME_PC)));
			if (op && op->type == R_ANAL_OP_TYPE_CALL) {
				until_addr = op->addr + op->size;
			}
			r_core_esil_step (core, until_addr, until_expr, NULL);
			r_anal_op_free (op);
			r_core_cmd0 (core, ".ar*");
			break;
		case 'p': //"aesp"
			n = strchr (input, ' ');
			n1 = n ? strchr (n + 1, ' ') : NULL;
			if ((!n || !n1) || (!(n + 1) || !(n1 + 1))) {
				eprintf ("aesp [offset] [num]\n");
				break;
			}
			adr = r_num_math (core->num, n + 1);
			off = r_num_math (core->num, n1 + 1);
			cmd_aespc (core, adr, off);
			break;
		case ' ':
			n = strchr (input, ' ');
			if (!(n + 1)) {
				r_core_esil_step (core, until_addr, until_expr, NULL);
				break;
			}
			off = r_num_math (core->num, n + 1);
			cmd_aespc (core, -1, off);
			break;
		default:
			r_core_esil_step (core, until_addr, until_expr, NULL);
			r_core_cmd0 (core, ".ar*");
			break;
		}
		break;
	case 'c': // "aec"
		if (input[1] == '?') { // "aec?"
			r_core_cmd_help (core, help_msg_aec);
		} else if (input[1] == 's') { // "aecs"
			const char *pc = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
			ut64 newaddr;
			int ret;
			for (;;) {
				op = r_core_anal_op (core, addr);
				if (!op) {
					break;
				}
				if (op->type == R_ANAL_OP_TYPE_SWI) {
					eprintf ("syscall at 0x%08" PFMT64x "\n", addr);
					break;
				}
				if (op->type == R_ANAL_OP_TYPE_TRAP) {
					eprintf ("trap at 0x%08" PFMT64x "\n", addr);
					break;
				}
				ret = r_core_esil_step (core, UT64_MAX, NULL, NULL);
				r_anal_op_free (op);
				op = NULL;
				if (core->anal->esil->trap || core->anal->esil->trap_code) {
					break;
				}
				if (!ret)
					break;
				r_core_cmd0 (core, ".ar*");
				newaddr = r_num_get (core->num, pc);
				if (addr == newaddr) {
					addr++;
					break;
				} else {
					addr = newaddr;
				}
			}
			if (op) {
				r_anal_op_free (op);
			}
		} else {
			// "aec"  -> continue until ^C
			// "aecu" -> until address
			// "aecue" -> until esil expression
			if (input[1] == 'u' && input[2] == 'e')
				until_expr = input + 3;
			else if (input[1] == 'u')
				until_addr = r_num_math (core->num, input + 2);
			else until_expr = "0";
			r_core_esil_step (core, until_addr, until_expr, NULL);
			r_core_cmd0 (core, ".ar*");
		}
		break;
	case 'i': // "aei"
		switch (input[1]) {
		case 's':
		case 'm': // "aeim"
			cmd_esil_mem (core, input + 2);
			break;
		case 'p': // initialize pc = $$
			r_core_cmd0 (core, "ar PC=$$");
			break;
		case '?':
			cmd_esil_mem (core, "?");
			break;
		case '-':
			if (esil) {
				sdb_reset (esil->stats);
			}
			r_anal_esil_free (esil);
			core->anal->esil = NULL;
			break;
		case 0:				//lolololol
			r_anal_esil_free (esil);
			// reinitialize
			{
				const char *pc = r_reg_get_name (core->anal->reg, R_REG_NAME_PC);
				if (r_reg_getv (core->anal->reg, pc) == 0LL) {
					r_core_cmd0 (core, "ar PC=$$");
				}
			}
			if (!(esil = core->anal->esil = r_anal_esil_new (stacksize, iotrap, addrsize))) {
				return;
			}
			r_anal_esil_setup (esil, core->anal, romem, stats, noNULL); // setup io
			esil->verbose = (int)r_config_get_i (core->config, "esil.verbose");
			/* restore user settings for interrupt handling */
			{
				const char *s = r_config_get (core->config, "cmd.esil.intr");
				if (s) {
					char *my = strdup (s);
					if (my) {
						r_config_set (core->config, "cmd.esil.intr", my);
						free (my);
					}
				}
			}
			break;
		}
		break;
	case 'k': // "aek"
		switch (input[1]) {
		case '\0':
			input = "123*";
			/* fall through */
		case ' ':
			if (esil && esil->stats) {
				char *out = sdb_querys (esil->stats, NULL, 0, input + 2);
				if (out) {
					r_cons_println (out);
					free (out);
				}
			} else {
				eprintf ("esil.stats is empty. Run 'aei'\n");
			}
			break;
		case '-':
			if (esil) {
				sdb_reset (esil->stats);
			}
			break;
		}
		break;
	case 'f': // "aef"
	{
		RListIter *iter;
		RAnalBlock *bb;
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal,
							core->offset, R_ANAL_FCN_TYPE_FCN | R_ANAL_FCN_TYPE_SYM);
		if (fcn) {
			// emulate every instruction in the function recursively across all the basic blocks
			r_list_foreach (fcn->bbs, iter, bb) {
				ut64 pc = bb->addr;
				ut64 end = bb->addr + bb->size;
				RAnalOp op;
				ut8 *buf;
				int ret, bbs = end - pc;
				if (bbs < 1 || bbs > 0xfffff) {
					eprintf ("Invalid block size\n");
				}
		//		eprintf ("[*] Emulating 0x%08"PFMT64x" basic block 0x%08" PFMT64x " - 0x%08" PFMT64x "\r[", fcn->addr, pc, end);
				buf = calloc (1, bbs + 1);
				r_io_read_at (core->io, pc, buf, bbs);
				int left;
				while (pc < end) {
					left = R_MIN (end - pc, 32);
					r_asm_set_pc (core->assembler, pc);
					ret = r_anal_op (core->anal, &op, addr, buf, left, R_ANAL_OP_MASK_ALL); // read overflow
					if (ret) {
						r_reg_set_value_by_role (core->anal->reg, R_REG_NAME_PC, pc);
						r_anal_esil_parse (esil, R_STRBUF_SAFEGET (&op.esil));
						r_anal_esil_dumpstack (esil);
						r_anal_esil_stack_free (esil);
						pc += op.size;
					} else {
						pc += 4; // XXX
					}
				}
			}
		} else {
			eprintf ("Cannot find function at 0x%08" PFMT64x "\n", core->offset);
		}
	} break;
	case 't': // "aet"
		switch (input[1]) {
		case 'r': // "aetr"
		{
			// anal ESIL to REIL.
			RAnalEsil *esil = r_anal_esil_new (stacksize, iotrap, addrsize);
			if (!esil)
				return;
			r_anal_esil_to_reil_setup (esil, core->anal, romem, stats);
			r_anal_esil_set_pc (esil, core->offset);
			r_anal_esil_parse (esil, input + 2);
			r_anal_esil_dumpstack (esil);
			r_anal_esil_free (esil);
			break;
		}
		case 's': // "aets"
			switch (input[2]) {
			case 0:
				r_anal_esil_session_list (esil);
				break;
			case '+':
				r_anal_esil_session_add (esil);
				break;
			default:
				r_core_cmd_help (core, help_msg_aets);
				break;
			}
			break;
		default:
			eprintf ("Unknown command. Use `aetr`.\n");
			break;
		}
		break;
	case 'A': // "aeA"
		if (input[1] == '?') {
			r_core_cmd_help (core, help_msg_aea);
		} else if (input[1] == 'r') {
			cmd_aea (core, 1 + (1<<1), core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'w') {
			cmd_aea (core, 1 + (1<<2), core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'n') {
			cmd_aea (core, 1 + (1<<3), core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'j') {
			cmd_aea (core, 1 + (1<<4), core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == '*') {
			cmd_aea (core, 1 + (1<<5), core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'f') {
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
			if (fcn) {
				cmd_aea (core, 1, fcn->addr, r_anal_fcn_size (fcn));
			}
		} else {
			cmd_aea (core, 1, core->offset, (int)r_num_math (core->num, input+2));
		}
		break;
	case 'a': // "aea"
		if (input[1] == '?') {
			r_core_cmd_help (core, help_msg_aea);
		} else if (input[1] == 'r') {
			cmd_aea (core, 1<<1, core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'w') {
			cmd_aea (core, 1<<2, core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'n') {
			cmd_aea (core, 1<<3, core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'j') {
			cmd_aea (core, 1<<4, core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == '*') {
			cmd_aea (core, 1<<5, core->offset, r_num_math (core->num, input+2));
		} else if (input[1] == 'f') {
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
                        // "aeafj"
			if (fcn) {
				switch (input[2]) {
				case 'j': // "aeafj"
					cmd_aea (core, 1<<4, fcn->addr, r_anal_fcn_size (fcn));
					break;
				default:
					cmd_aea (core, 1, fcn->addr, r_anal_fcn_size (fcn));
					break;
				}
				break;
			}
		} else {
			const char *arg = input[1]? input + 2: "";
			ut64 len = r_num_math (core->num, arg);
			cmd_aea (core, 0, core->offset, len);
		}
		break;
	case 'x': { // "aex"
		char *hex;
		int ret, bufsz;

		input = r_str_trim_ro (input + 1);
		hex = strdup (input);
		if (!hex) {
			break;
		}

		RAnalOp aop = R_EMPTY;
		bufsz = r_hex_str2bin (hex, (ut8*)hex);
		ret = r_anal_op (core->anal, &aop, core->offset,
			(const ut8*)hex, bufsz, R_ANAL_OP_MASK_ALL);
		if (ret>0) {
			const char *str = R_STRBUF_SAFEGET (&aop.esil);
			char *str2 = r_str_newf (" %s", str);
			cmd_anal_esil (core, str2);
			free (str2);
		}
		r_anal_op_fini (&aop);
		break;
	}
	case '?': // "ae?"
		if (input[1] == '?') {
			r_core_cmd_help (core, help_detail_ae);
			break;
		}
		/* fallthrough */
	default:
		r_core_cmd_help (core, help_msg_ae);
		break;
	}
}

static void cmd_anal_bytes(RCore *core, const char *input) {
	int len = core->blocksize;
	int tbs = len;
	if (input[0]) {
		len = (int)r_num_get (core->num, input + 1);
		if (len > tbs) {
			r_core_block_size (core, len);
		}
	}
	core_anal_bytes (core, core->block, len, 0, input[0]);
	if (tbs != core->blocksize) {
		r_core_block_size (core, tbs);
	}
}

static void cmd_anal_opcode(RCore *core, const char *input) {
	int l, len = core->blocksize;
	ut32 tbs = core->blocksize;

	switch (input[0]) {
	case '?':
		r_core_cmd_help (core, help_msg_ao);
		break;
	case 's': // "aos"
	case 'j': // "aoj"
	case 'e': // "aoe"
	case 'r': {
		int count = 1;
		if (input[1] && input[2]) {
			l = (int)r_num_get (core->num, input + 1);
			if (l > 0) {
				count = l;
			}
			if (l > tbs) {
				r_core_block_size (core, l * 4);
			//	len = l;
			}
		} else {
			len = l = core->blocksize;
			count = 1;
		}
		core_anal_bytes (core, core->block, len, count, input[0]);
		}
		break;
	case '*':
		r_core_anal_hint_list (core->anal, input[0]);
		break;
	default: {
		int count = 0;
		if (input[0]) {
			l = (int)r_num_get (core->num, input + 1);
			if (l > 0) {
				count = l;
			}
			if (l > tbs) {
				r_core_block_size (core, l * 4);
				//len = l;
			}
		} else {
			len = l = core->blocksize;
			count = 1;
		}
		core_anal_bytes (core, core->block, len, count, 0);
		break;
	}
	}
}

static void cmd_anal_jumps(RCore *core, const char *input) {
	r_core_cmdf (core, "af @@= `ax~ref.code.jmp[1]`");
}

// TODO: cleanup to reuse code
static void cmd_anal_aftertraps(RCore *core, const char *input) {
	int bufi, minop = 1; // 4
	ut8 *buf;
	RBinFile *binfile;
	RAnalOp op;
	ut64 addr, addr_end;
	ut64 len = r_num_math (core->num, input);
	if (len > 0xffffff) {
		eprintf ("Too big\n");
		return;
	}
	binfile = r_core_bin_cur (core);
	if (!binfile) {
		eprintf ("cur binfile NULL\n");
		return;
	}
	addr = core->offset;
	if (!len) {
		// ignore search.in to avoid problems. analysis != search
		RIOSection *sec = r_io_section_vget (core->io, addr);
		if (sec && sec->flags & 1) {
			// search in current section
			if (sec->size > binfile->size) {
				addr = sec->vaddr;
				if (binfile->size > sec->paddr) {
					len = binfile->size - sec->paddr;
				} else {
					eprintf ("Opps something went wrong aac\n");
					return;
				}
			} else {
				addr = sec->vaddr;
				len = sec->size;
			}
		} else {
			if (sec && sec->vaddr != sec->paddr && binfile->size > (core->offset - sec->vaddr + sec->paddr)) {
				len = binfile->size - (core->offset - sec->vaddr + sec->paddr);
			} else {
				if (binfile->size > core->offset) {
					len = binfile->size - core->offset;
				} else {
					eprintf ("Oops invalid range\n");
					len = 0;
				}
			}
		}
	}
	addr_end = addr + len;
	if (!(buf = malloc (4096))) {
		return;
	}
	bufi = 0;
	int trapcount = 0;
	int nopcount = 0;
	r_cons_break_push (NULL, NULL);
	while (addr < addr_end) {
		if (r_cons_is_breaked ()) {
			break;
		}
		// TODO: too many ioreads here
		if (bufi > 4000) {
			bufi = 0;
		}
		if (!bufi) {
			r_io_read_at (core->io, addr, buf, 4096);
		}
		if (r_anal_op (core->anal, &op, addr, buf + bufi, 4096 - bufi, R_ANAL_OP_MASK_ALL)) {
			if (op.size < 1) {
				// XXX must be +4 on arm/mips/.. like we do in disasm.c
				op.size = minop;
			}
			if (op.type == R_ANAL_OP_TYPE_TRAP) {
				trapcount ++;
			} else if (op.type == R_ANAL_OP_TYPE_NOP) {
				nopcount ++;
			} else {
				if (nopcount > 1) {
					r_cons_printf ("af @ 0x%08"PFMT64x"\n", addr);
					nopcount = 0;
				}
				if (trapcount > 0) {
					r_cons_printf ("af @ 0x%08"PFMT64x"\n", addr);
					trapcount = 0;
				}
			}
		} else {
			op.size = minop;
		}
		addr += (op.size > 0)? op.size : 1;
		bufi += (op.size > 0)? op.size : 1;
		r_anal_op_fini (&op);
	}
	r_cons_break_pop ();
	free (buf);
}

static void cmd_anal_blocks(RCore *core, const char *input) {
	ut64 from , to;
	char *arg = strchr (input, ' ');
	r_cons_break_push (NULL, NULL);
#if 0
	ls_foreach (core->io->sections, iter, s) {
		/* is executable */
		if (!(s->flags & R_IO_EXEC)) {
			continue;
		}
		min = s->vaddr;
		max = s->vaddr + s->vsize;
		r_core_cmdf (core, "abb%s 0x%08"PFMT64x" @ 0x%08"PFMT64x, input, (max - min), min);
		if (r_cons_is_breaked ()) {
			goto ctrl_c;
		}
	}
	if (ls_empty (core->io->sections)) {
		min = core->offset;
		max = 0xffff + min;
		r_core_cmdf (core, "abb%s 0x%08"PFMT64x" @ 0x%08"PFMT64x, input, (max - min), min);
		if (r_cons_is_breaked ()) {
			goto ctrl_c;
		}
	}
#endif
	if (!arg) {
		RList *list = r_core_get_boundaries_prot (core, R_IO_EXEC, NULL, "anal");
		RListIter *iter;
		RIOMap* map;
		r_list_foreach (list, iter, map) {
			from = map->itv.addr;
			to = r_itv_end (map->itv);
			if (r_cons_is_breaked ()) {
				goto ctrl_c;
			}
			if (!from && !to) {
				eprintf ("Cannot determine search boundaries\n");
			} else if (to - from > UT32_MAX) {
				eprintf ("Skipping huge range\n");
			} else {
				r_core_cmdf (core, "abb 0x%08"PFMT64x" @ 0x%08"PFMT64x, (to - from), from);
			}
		}
	} else {
		int sz = r_num_math (core->num, arg + 1);
		r_core_cmdf (core, "abb 0x%08"PFMT64x" @ 0x%08"PFMT64x, sz, core->offset);
	}
ctrl_c:
	r_cons_break_pop ();
}

static void _anal_calls(RCore *core, ut64 addr, ut64 addr_end) {
	RAnalOp op;
	int bufi;
	int depth = r_config_get_i (core->config, "anal.depth");
	const int addrbytes = core->io->addrbytes;
	const int bsz = 4096;
	ut8 *buf;
	ut8 *block;
	bufi = 0;
	if (addr_end - addr > UT32_MAX) {
		return;
	}
	buf = malloc (bsz);
	block = malloc (bsz);
	if (!buf || !block) {
		eprintf ("Error: cannot allocate buf or block\n");
		free (buf);
		free (block);
		return;
	}

	int minop = r_anal_archinfo (core->anal, R_ANAL_ARCHINFO_MIN_OP_SIZE);
	if (minop < 1) {
		minop = 1;
	}
	while (addr < addr_end) {
		if (r_cons_is_breaked ()) {
			break;
		}
		// TODO: too many ioreads here
		if (bufi > 4000) {
			bufi = 0;
		}
		if (!bufi) {
			r_io_read_at (core->io, addr, buf, bsz);
		}
		memset (block, -1, bsz);
		if (!memcmp (buf, block, bsz)) {
			//eprintf ("Error: skipping uninitialized block \n");
			addr += bsz;
			continue;
		}
		memset (block, 0, bsz);
		if (!memcmp (buf, block, bsz)) {
			//eprintf ("Error: skipping uninitialized block \n");
			addr += bsz;
			continue;
		}
		if (r_anal_op (core->anal, &op, addr, buf + bufi, bsz - bufi, 0) > 0) {
			if (op.size < 1) {
				op.size = minop;
			}
			if (op.type == R_ANAL_OP_TYPE_CALL) {
#if JAYRO_03
#error FUCK
				if (!anal_is_bad_call (core, from, to, addr, buf, bufi)) {
					fcn = r_anal_get_fcn_in (core->anal, op.jump, R_ANAL_FCN_TYPE_ROOT);
					if (!fcn) {
						r_core_anal_fcn (core, op.jump, addr,
						  R_ANAL_REF_TYPE_NULL, depth);
					}
				}
#else
				// add xref here
				r_anal_xrefs_set (core->anal, R_ANAL_REF_TYPE_CALL, addr, op.jump);
				if (r_io_is_valid_offset (core->io, op.jump, 1)) {
					r_core_anal_fcn (core, op.jump, addr, R_ANAL_REF_TYPE_NULL, depth);
				}
#endif
			}

		} else {
			op.size = minop;
		}
		if ((int)op.size < 1) {
			op.size = minop;
		}
		addr += op.size;
		bufi += addrbytes * op.size;
		r_anal_op_fini (&op);
	}
	free (buf);
	free (block);
}

static void cmd_anal_calls(RCore *core, const char *input, bool only_print_flag) {
	RList *ranges = NULL;
	RIOMap *r;
	RBinFile *binfile;
	ut64 addr;
	ut64 len = r_num_math (core->num, input);
	if (len > 0xffffff) {
		eprintf ("Too big\n");
		return;
	}
	binfile = r_core_bin_cur (core);
	addr = core->offset;
	if (binfile) {
		if (len) {
			RIOMap *m = R_NEW0 (RIOMap);
			m->itv.addr = addr;
			m->itv.size = len;
			r_list_append (ranges, m);
		} else {
			ranges = r_core_get_boundaries_prot (core, R_IO_EXEC, NULL, "anal");
		}
	}
	r_cons_break_push (NULL, NULL);
	if (!binfile || !r_list_length (ranges)) {
		RListIter *iter;
		RIOMap *map;
		r_list_free (ranges);
		ranges = r_core_get_boundaries_prot (core, 0, NULL, "anal");
		r_list_foreach (ranges, iter, map) {
			ut64 addr = map->itv.addr;
			if (only_print_flag) {
				r_cons_printf ("f fcn.0x%08"PFMT64x" %d 0x%08"PFMT64x"\n",
					addr, map->itv.size, addr);
			} else {
				_anal_calls (core, addr, r_itv_end (map->itv));
			}
		}
	} else {
		RListIter *iter;
		if (binfile) {
			r_list_foreach (ranges, iter, r) {
				addr = r->itv.addr;
				//this normally will happen on fuzzed binaries, dunno if with huge
				//binaries as well
				if (r_cons_is_breaked ()) {
					break;
				}
				if (only_print_flag) {
					r_cons_printf ("f fcn.0x%08"PFMT64x" %d 0x%08"PFMT64x"\n",
						addr, r->itv.size, addr);
				} else {
					_anal_calls (core, addr, r_itv_end (r->itv));
				}
			}
		}
	}
	r_cons_break_pop ();
	r_list_free (ranges);
}

static void cmd_asf(RCore *core, const char *input) {
	char *ret;
	if (input[0] == ' ') {
		ret = sdb_querys (core->anal->sdb_fcnsign, NULL, 0, input + 1);
	} else {
		ret = sdb_querys (core->anal->sdb_fcnsign, NULL, 0, "*");
	}
	if (ret && *ret) {
		r_cons_println (ret);
	}
	free (ret);
}

static void cmd_anal_syscall(RCore *core, const char *input) {
	RSyscallItem *si;
	RListIter *iter;
	RList *list;
	RNum *num = NULL;
	char *out;
	int n;

	switch (input[0]) {
	case 'c': // "asc"
		if (input[1] == 'a') {
			if (input[2] == ' ') {
				if (!isalpha (input[3]) && (n = r_num_math (num, input + 3)) >= 0 ) {
					si = r_syscall_get (core->anal->syscall, n, -1);
					if (si)
						r_cons_printf (".equ SYS_%s %d\n", si->name, n);
					else eprintf ("Unknown syscall number\n");
				} else {
					n = r_syscall_get_num (core->anal->syscall, input + 3);
					if (n != -1) {
						r_cons_printf (".equ SYS_%s %d\n", input + 3, n);
					} else {
						eprintf ("Unknown syscall name\n");
					}
				}
			} else {
				list = r_syscall_list (core->anal->syscall);
				r_list_foreach (list, iter, si) {
					r_cons_printf (".equ SYS_%s %d\n",
						si->name, (ut32)si->num);
				}
				r_list_free (list);
			}
		} else {
			if (input[1] == ' ') {
				if (!isalpha (input[2]) && (n = r_num_math (num, input + 2)) >= 0 ) {
					si = r_syscall_get (core->anal->syscall, n, -1);
					if (si)
						r_cons_printf ("#define SYS_%s %d\n", si->name, n);
					else eprintf ("Unknown syscall number\n");
				} else {
					n = r_syscall_get_num (core->anal->syscall, input + 2);
					if (n != -1) {
						r_cons_printf ("#define SYS_%s %d\n", input + 2, n);
					} else {
						eprintf ("Unknown syscall name\n");
					}
				}
			} else {
				list = r_syscall_list (core->anal->syscall);
				r_list_foreach (list, iter, si) {
					r_cons_printf ("#define SYS_%s %d\n",
						si->name, (ut32)si->num);
				}
				r_list_free (list);
			}
		}
		break;
	case 'f': // "asf"
		cmd_asf (core, input + 1);
		break;
	case 'l': // "asl"
		if (input[1] == ' ') {
			if (!isalpha (input[2]) && (n = r_num_math (num, input + 2)) >= 0 ) {
				si = r_syscall_get (core->anal->syscall, n, -1);
				if (si)
					r_cons_println (si->name);
				else eprintf ("Unknown syscall number\n");
			} else {
				n = r_syscall_get_num (core->anal->syscall, input + 2);
				if (n != -1) {
					r_cons_printf ("%d\n", n);
				} else {
					eprintf ("Unknown syscall name\n");
				}
			}
		} else {
			list = r_syscall_list (core->anal->syscall);
			r_list_foreach (list, iter, si) {
				r_cons_printf ("%s = 0x%02x.%u\n",
					si->name, (ut32)si->swi, (ut32)si->num);
			}
			r_list_free (list);
		}
		break;
	case 'j': // "asj"
		list = r_syscall_list (core->anal->syscall);
		r_cons_printf ("[");
		r_list_foreach (list, iter, si) {
			r_cons_printf ("{\"name\":\"%s\","
				"\"swi\":\"%d\",\"num\":\"%d\"}",
				si->name, si->swi, si->num);
			if (iter->n) {
				r_cons_printf (",");
			}
		}
		r_cons_printf ("]\n");
		r_list_free (list);
		// JSON support
		break;
	case '\0':
		cmd_syscall_do (core, -1); //n);
		break;
	case ' ':
		cmd_syscall_do (core, (int)r_num_get (core->num, input + 1));
		break;
	case 'k': // "ask"
		if (input[1] == ' ') {
			out = sdb_querys (core->anal->syscall->db, NULL, 0, input + 2);
			if (out) {
				r_cons_println (out);
				free (out);
			}
		} else eprintf ("|ERROR| Usage: ask [query]\n");
		break;
	default:
	case '?':
		r_core_cmd_help (core, help_msg_as);
		break;
	}
}

static void anal_axg (RCore *core, const char *input, int level, Sdb *db, int opts) {
	char arg[32], pre[128];
	RList *xrefs;
	RListIter *iter;
	RAnalRef *ref;
	ut64 addr = core->offset;
	int is_json = opts & R_CORE_ANAL_JSON;
	if (input && *input) {
		addr = r_num_math (core->num, input);
	}
	int spaces = (level + 1) * 2;
	if (spaces > sizeof (pre) - 4) {
		spaces = sizeof (pre) - 4;
	}
	memset (pre, ' ', sizeof (pre));
	strcpy (pre+spaces, "- ");

	xrefs = r_anal_xrefs_get (core->anal, addr);
	if (!r_list_empty (xrefs)) {
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, -1);
		if (fcn) {
			if (is_json) {
				r_cons_printf ("{\"%"PFMT64d"\":{\"type\":\"fcn\","
					"\"fcn_addr\":%"PFMT64d",\"name\":\"%s\",\"refs\":[",
					addr, fcn->addr, fcn->name);
			} else {
				//if (sdb_add (db, fcn->name, "1", 0)) {
				r_cons_printf ("%s0x%08"PFMT64x" fcn 0x%08"PFMT64x" %s\n",
					pre + 2, addr, fcn->addr, fcn->name);
				//}
			}
		} else {
			if (is_json) {
				r_cons_printf ("{\"%"PFMT64d"\":{\"refs\":[", addr);
			} else {
			//snprintf (arg, sizeof (arg), "0x%08"PFMT64x, addr);
			//if (sdb_add (db, arg, "1", 0)) {
				r_cons_printf ("%s0x%08"PFMT64x"\n", pre+2, addr);
			//}
			}
		}
	}
	r_list_foreach (xrefs, iter, ref) {
		RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, ref->addr, -1);
		if (fcn) {
			if (is_json) {
				if (level == 0) {
					r_cons_printf ("{\"%"PFMT64d"\":{\"type\":\"fcn\",\"fcn_addr\": %"PFMT64d",\"name\":\"%s\",\"refs\":[", ref->addr, fcn->addr, fcn->name);
				} else {
					r_cons_printf ("]}},{\"%"PFMT64d"\":{\"type\":\"fcn\",\"fcn_addr\": %"PFMT64d",\"name\":\"%s\",\"refs\":[", ref->addr, fcn->addr, fcn->name);

				}
			} else {
				r_cons_printf ("%s0x%08"PFMT64x" fcn 0x%08"PFMT64x" %s\n", pre, ref->addr, fcn->addr, fcn->name);
			}
			if (sdb_add (db, fcn->name, "1", 0)) {
				snprintf (arg, sizeof (arg), "0x%08"PFMT64x, fcn->addr);
				anal_axg (core, arg, level+1, db, opts);
			} else {
				if (is_json) {
					r_cons_printf("]}}");
				}
			}
			if (is_json) {
				if (iter->n) {
					r_cons_printf (",");
				}
			}
		} else {
			if (is_json) {
				r_cons_printf ("{\"%"PFMT64d"\":{\"type\":\"???\",\"refs\":[", ref->addr);
			} else {
				r_cons_printf ("%s0x%08"PFMT64x" ???\n", pre, ref->addr);
			}
			snprintf (arg, sizeof (arg), "0x%08"PFMT64x, ref->addr);
			if (sdb_add (db, arg, "1", 0)) {
				anal_axg (core, arg, level +1, db, opts);
			} else {
				if (is_json) {
					r_cons_printf("]}}");
				}
			}
			if (is_json) {
				if (iter->n) {
					r_cons_printf (",");
				}
			}
		}
	}
	if (is_json) {
		r_cons_printf("]}}");
		if (level == 0) {
			r_cons_printf("\n");
		}
	}
	r_list_free (xrefs);
}

static void cmd_anal_ucall_ref (RCore *core, ut64 addr) {
	RAnalFunction * fcn = r_anal_get_fcn_at (core->anal, addr, R_ANAL_FCN_TYPE_NULL);
	if (fcn) {
		r_cons_printf (" ; %s", fcn->name);
	} else {
		r_cons_printf (" ; 0x%" PFMT64x, addr);
	}
}

static bool cmd_anal_refs(RCore *core, const char *input) {
	ut64 addr = core->offset;
	switch (input[0]) {
	case '-': { // "ax-"
		RList *list;
		RListIter *iter;
		RAnalRef *ref;
		char *cp_inp = strdup (input + 1);
		char *ptr = r_str_trim_head (cp_inp);
		if (!strcmp (ptr, "*")) {
			r_anal_xrefs_init (core->anal);
		} else {
			int n = r_str_word_set0 (ptr);
			ut64 from = UT64_MAX, to = UT64_MAX;
			switch (n) {
			case 2:
				from = r_num_math (core->num, r_str_word_get0 (ptr, 1));
				//fall through
			case 1: // get addr
				to = r_num_math (core->num, r_str_word_get0 (ptr, 0));
				break;
			default:
				to = core->offset;
				break;
			}
			list = r_anal_xrefs_get (core->anal, to);
			if (list) {
				r_list_foreach (list, iter, ref) {
					if (from != UT64_MAX && from == ref->addr) {
						r_anal_ref_del (core->anal, ref->addr, ref->at);
					}
					if (from == UT64_MAX) {
						r_anal_ref_del (core->anal, ref->addr, ref->at);
					}
				}
				r_list_free (list);
			}
		}
		free (cp_inp);
	} break;
	case 'g': // "axg"
		{
			Sdb *db = sdb_new0 ();
			if(input[1] == '\0') {
				anal_axg (core, input[1] ? input + 2 : NULL, 0, db, 0);
			} else if(input[1] == 'j') {
				anal_axg (core, input[1] ? input + 2 : NULL, 0, db, R_CORE_ANAL_JSON);
			}
			sdb_free (db);
		}
		break;
	case 'k': // "axk"
		if (input[1] == '?') {
			eprintf ("Usage: axk [query]\n");
		} else if (input[1] == ' ') {
			sdb_query (core->anal->sdb_xrefs, input + 2);
		} else {
			r_core_anal_ref_list (core, 'k');
		}
		break;
	case '\0': // "ax"
	case 'j': // "axj"
	case 'q': // "axq"
	case '*': // "ax*"
		r_core_anal_ref_list (core, input[0]);
		break;
	case 't': { // "axt"
		const int size = 12;
		RList *list;
		RAnalFunction *fcn;
		RAnalRef *ref;
		RListIter *iter;
		ut8 buf[12];
		RAsmOp asmop;
		char *buf_asm = NULL;
		char *space = strchr (input, ' ');

		if (space) {
			addr = r_num_math (core->num, space + 1);
		} else {
			addr = core->offset;
		}
		list = r_anal_xrefs_get (core->anal, addr);
		if (list) {
			if (input[1] == 'q') { // "axtq"
				r_list_foreach (list, iter, ref) {
					r_cons_printf ("0x%" PFMT64x "\n", ref->addr);
				}
			} else if (input[1] == 'j') { // "axtj"
				bool asm_varsub = r_config_get_i (core->config, "asm.varsub");
				core->parser->relsub = r_config_get_i (core->config, "asm.relsub");
				core->parser->localvar_only = r_config_get_i (core->config, "asm.varsub_only");
				r_cons_printf ("[");
				r_list_foreach (list, iter, ref) {
					r_core_read_at (core, ref->addr, buf, size);
					r_asm_set_pc (core->assembler, ref->addr);
					r_asm_disassemble (core->assembler, &asmop, buf, size);
					char str[512];
					fcn = r_anal_get_fcn_in (core->anal, ref->addr, 0);
					if (asm_varsub) {
						r_parse_varsub (core->parser, fcn, ref->addr, asmop.size,
								asmop.buf_asm, asmop.buf_asm, sizeof (asmop.buf_asm));
					}
					r_parse_filter (core->parser, core->flags,
							asmop.buf_asm, str, sizeof (str), core->print->big_endian);

					r_cons_printf ("{\"from\":%" PFMT64u ",\"type\":\"%s\",\"opcode\":\"%s\"", ref->addr, r_anal_ref_to_string (ref->type), str);
					if (fcn) {
						r_cons_printf (",\"fcn_addr\":%"PFMT64d",\"fcn_name\":\"%s\"", fcn->addr, fcn->name);
					}
					RFlagItem *fi = r_flag_get_at (core->flags, fcn? fcn->addr: ref->addr, true);
					if (fi) {
						if (fcn && strcmp (fcn->name, fi->name)) {
							r_cons_printf (",\"flag\":\"%s\"", fi->name);
						}
						if (fi->realname && strcmp (fi->name, fi->realname)) {
							r_cons_printf (",\"realname\":\"%s\"", fi->realname);
						}
					}
					r_cons_printf ("}%s", iter->n? ",": "");
				}
				r_cons_printf ("]");
				r_cons_newline ();
			} else if (input[1] == 'g') { // axtg
				r_list_foreach (list, iter, ref) {
					char *str = r_core_cmd_strf (core, "fd 0x%"PFMT64x, ref->addr);
					if (!str) {
						str = strdup ("?\n");
					}
					r_str_trim_tail (str);
					r_cons_printf ("agn 0x%" PFMT64x " \"%s\"\n", ref->addr, str);
					free (str);
				}
				if (input[2] != '*') {
					RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, 0);
					r_cons_printf ("agn 0x%" PFMT64x " \"%s\"\n", addr, fcn?fcn->name: "$$");
				}
				r_list_foreach (list, iter, ref) {
					r_cons_printf ("age 0x%" PFMT64x " 0x%"PFMT64x"\n", ref->addr, addr);
				}
			} else if (input[1] == '*') { // axt*
				// TODO: implement multi-line comments
				r_list_foreach (list, iter, ref)
					r_cons_printf ("CCa 0x%" PFMT64x " \"XREF type %d at 0x%" PFMT64x"%s\n",
						ref->addr, ref->type, addr, iter->n? ",": "");
			} else { // axt
				int has_color = core->print->flags & R_PRINT_FLAGS_COLOR;
				char str[512];
				RAnalFunction *fcn;
				char *comment;
				bool asm_varsub = r_config_get_i (core->config, "asm.varsub");
				core->parser->relsub = r_config_get_i (core->config, "asm.relsub");
				core->parser->localvar_only = r_config_get_i (core->config, "asm.varsub_only");
				if (core->parser->relsub) {
					core->parser->relsub_addr = addr;
				}
				r_list_foreach (list, iter, ref) {
					r_core_read_at (core, ref->addr, buf, size);
					r_asm_set_pc (core->assembler, ref->addr);
					r_asm_disassemble (core->assembler, &asmop, buf, size);

					fcn = r_anal_get_fcn_in (core->anal, ref->addr, 0);
					if (asm_varsub) {
						r_parse_varsub (core->parser, fcn, ref->addr, asmop.size,
								asmop.buf_asm, asmop.buf_asm, sizeof (asmop.buf_asm));
					}
					r_parse_filter (core->parser, core->flags,
							asmop.buf_asm, str, sizeof (str), core->print->big_endian);
					if (has_color) {
						buf_asm = r_print_colorize_opcode (core->print, str,
							core->cons->pal.reg, core->cons->pal.num, false);
					} else {
						buf_asm = r_str_new (str);
					}
					comment = r_meta_get_string (core->anal, R_META_TYPE_COMMENT, ref->addr);
					char *buf_fcn = comment
						? r_str_newf ("%s; %s", fcn ?  fcn->name : "(nofunc)", strtok (comment, "\n"))
						: r_str_newf ("%s", fcn ? fcn->name : "(nofunc)");
					r_cons_printf ("%s 0x%" PFMT64x " [%s] %s\n",
						buf_fcn, ref->addr, r_anal_ref_to_string (ref->type), buf_asm);
					free (buf_asm);
					free (buf_fcn);
				}
			}
			r_list_free (list);
		} else {
			if (input[1] == 'j') { // "axtj"
				r_cons_print ("[]\n");
			}
		}
	} break;
	case 'f': { // "axf"
		ut8 buf[12];
		RAsmOp asmop;
		char *buf_asm = NULL;
		RList *list, *list_ = NULL;
		RAnalRef *ref;
		RListIter *iter;
		char *space = strchr (input, ' ');

		if (space) {
			addr = r_num_math (core->num, space + 1);
		} else {
			addr = core->offset;
		}
		if (input[1] == '.') { // axf.
			list = list_ = r_anal_xrefs_get_from (core->anal, addr);
			if (!list) {
				RAnalFunction * fcn = r_anal_get_fcn_in (core->anal, addr, 0);
				list = r_anal_fcn_get_refs (core->anal, fcn);
			}
		} else {
			list = r_anal_refs_get (core->anal, addr);
		}

		if (list) {
			if (input[1] == 'q') { // axfq
				r_list_foreach (list, iter, ref) {
					r_cons_printf ("0x%" PFMT64x "\n", ref->at);
				}
			} else if (input[1] == 'j') { // axfj
				r_cons_print ("[");
				r_list_foreach (list, iter, ref) {
					r_core_read_at (core, ref->at, buf, 12);
					r_asm_set_pc (core->assembler, ref->at);
					r_asm_disassemble (core->assembler, &asmop, buf, 12);
					r_cons_printf ("{\"from\":%" PFMT64d ",\"to\":%" PFMT64d ",\"type\":\"%s\",\"opcode\":\"%s\"}%s",
						ref->at, ref->addr, r_anal_ref_to_string (ref->type), asmop.buf_asm, iter->n? ",": "");
				}
				r_cons_print ("]\n");
			} else if (input[1] == '*') { // axf*
				// TODO: implement multi-line comments
				r_list_foreach (list, iter, ref) {
					r_cons_printf ("CCa 0x%" PFMT64x " \"XREF from 0x%" PFMT64x "\n",
						ref->at, ref->type, asmop.buf_asm, iter->n? ",": "");
				}
			} else { // axf
				char str[512];
				int has_color = core->print->flags & R_PRINT_FLAGS_COLOR;
				r_list_foreach (list, iter, ref) {
					r_core_read_at (core, ref->at, buf, 12);
					r_asm_set_pc (core->assembler, ref->at);
					r_asm_disassemble (core->assembler, &asmop, buf, 12);
					r_parse_filter (core->parser, core->flags,
							asmop.buf_asm, str, sizeof (str), core->print->big_endian);
					if (has_color) {
						buf_asm = r_print_colorize_opcode (core->print, str,
							core->cons->pal.reg, core->cons->pal.num, false);
					} else {
						buf_asm = r_str_new (str);
					}
					r_cons_printf ("%c 0x%" PFMT64x " %s",
						ref->type, ref->at, buf_asm);

					if (ref->type == R_ANAL_REF_TYPE_CALL) {
						RAnalOp aop;
						r_anal_op (core->anal, &aop, ref->at, buf, 12, R_ANAL_OP_MASK_ALL);
						if (aop.type == R_ANAL_OP_TYPE_UCALL) {
							cmd_anal_ucall_ref (core, ref->addr);
						}
					}
					r_cons_newline ();
					free (buf_asm);
				}
			}
			r_list_free (list_);
			r_list_free (list);
		} else {
			if (input[1] == 'j') { // axfj
				r_cons_print ("[]\n");
			}
		}
	} break;
	case 'F':
		find_refs (core, input + 1);
		break;
	case 'C': // "axC"
	case 'c': // "axc"
	case 'd': // "axd"
	case ' ': // "ax "
		{
		char *ptr = strdup (r_str_trim_head ((char *)input + 1));
		int n = r_str_word_set0 (ptr);
		ut64 at = core->offset;
		ut64 addr = UT64_MAX;
		switch (n) {
		case 2: // get at
			at = r_num_math (core->num, r_str_word_get0 (ptr, 1));
		/* fall through */
		case 1: // get addr
			addr = r_num_math (core->num, r_str_word_get0 (ptr, 0));
			break;
		default:
			free (ptr);
			return false;
		}
		r_anal_xrefs_set (core->anal, input[0], at, addr);
		free (ptr);
		}
	   	break;
	default:
	case '?':
		r_core_cmd_help (core, help_msg_ax);
		break;
	}

	return true;
}
static void cmd_anal_hint(RCore *core, const char *input) {
	switch (input[0]) {
	case '?':
		if (input[1]) {
			ut64 addr = r_num_math (core->num, input + 1);
			r_core_anal_hint_print (core->anal, addr, 0);
		} else {
			r_core_cmd_help (core, help_msg_ah);
		}
		break;
	case '.': // "ah."
		r_core_anal_hint_print (core->anal, core->offset, 0);
		break;
	case 'a': // "aha" set arch
		if (input[1]) {
			int i;
			char *ptr = strdup (input + 2);
			i = r_str_word_set0 (ptr);
			if (i == 2) {
				r_num_math (core->num, r_str_word_get0 (ptr, 1));
			}
			r_anal_hint_set_arch (core->anal, core->offset, r_str_word_get0 (ptr, 0));
			free (ptr);
		} else if (input[1] == '-') {
			r_anal_hint_unset_arch (core->anal, core->offset);
		} else {
			eprintf ("Missing argument\n");
		}
		break;
	case 'b': // "ahb" set bits
		if (input[1]) {
			char *ptr = strdup (input + 2);
			int bits;
			int i = r_str_word_set0 (ptr);
			if (i == 2) {
				r_num_math (core->num, r_str_word_get0 (ptr, 1));
			}
			bits = r_num_math (core->num, r_str_word_get0 (ptr, 0));
			r_anal_hint_set_bits (core->anal, core->offset, bits);
			free (ptr);
		}  else if (input[1] == '-') {
			r_anal_hint_unset_bits (core->anal, core->offset);
		} else {
			eprintf ("Missing argument\n");
		}
		break;
	case 'i': // "ahi"
		if (input[1] == '?') {
			r_core_cmd_help (core, help_msg_ahi);
		} else if (input[1] == ' ') {
		// You can either specify immbase with letters, or numbers
			const int base =
				(input[2] == 's') ? 1 :
				(input[2] == 'b') ? 2 :
				(input[2] == 'p') ? 3 :
				(input[2] == 'o') ? 8 :
				(input[2] == 'd') ? 10 :
				(input[2] == 'h') ? 16 :
				(input[2] == 'i') ? 32 : // ip address
				(input[2] == 'S') ? 80 : // syscall
				(int) r_num_math (core->num, input + 1);
			r_anal_hint_set_immbase (core->anal, core->offset, base);
		} else if (input[1] == '-') { // "ahi-"
			r_anal_hint_set_immbase (core->anal, core->offset, 0);
		} else {
			eprintf ("|ERROR| Usage: ahi [base]\n");
		}
		break;
	case 'h': // "ahh"
		if (input[1] == '-') {
			r_anal_hint_unset_high (core->anal, core->offset);
		} else if (input[1] == ' ') {
			r_anal_hint_set_high (core->anal, r_num_math (core->num, input + 1));
		} else {
			r_anal_hint_set_high (core->anal, core->offset);
		}
		break;
	case 'c': // "ahc"
		if (input[1] == ' ') {
			r_anal_hint_set_jump (
				core->anal, core->offset,
				r_num_math (core->num, input + 1));
		} else if (input[1] == '-') {
			r_anal_hint_unset_jump (core->anal, core->offset);
		}
		break;
	case 'f': // "ahf"
		if (input[1] == ' ') {
			r_anal_hint_set_fail (
				core->anal, core->offset,
				r_num_math (core->num, input + 1));
		} else if (input[1] == '-') {
			r_anal_hint_unset_fail (core->anal, core->offset);
		}
		break;
	case 's': // "ahs" set size (opcode length)
		if (input[1] == ' ') {
			r_anal_hint_set_size (core->anal, core->offset, atoi (input + 1));
		} else if (input[1] == '-') {
			r_anal_hint_unset_size (core->anal, core->offset);
		} else {
			eprintf ("Usage: ahs 16\n");
		}
		break;
	case 'S': // "ahS" set size (opcode length)
		if (input[1] == ' ') {
			r_anal_hint_set_syntax (core->anal, core->offset, input + 2);
		} else if (input[1] == '-') {
			r_anal_hint_unset_syntax (core->anal, core->offset);
		} else {
			eprintf ("Usage: ahS att\n");
		}
		break;
	case 'o': // "aho" set opcode string
		if (input[1] == ' ') {
			r_anal_hint_set_opcode (core->anal, core->offset, input + 2);
		} else if (input[1] == '-') {
			r_anal_hint_unset_opcode (core->anal, core->offset);
		} else {
			eprintf ("Usage: aho popall\n");
		}
		break;
	case 'e': // "ahe" set ESIL string
		if (input[1] == ' ') {
			r_anal_hint_set_esil (core->anal, core->offset, input + 2);
		} else if (input[1] == '-') {
			r_anal_hint_unset_esil (core->anal, core->offset);
		} else {
			eprintf ("Usage: ahe r0,pc,=\n");
		}
		break;
#if 0
	case 'e': // set endian
		if (input[1] == ' ') {
			r_anal_hint_set_opcode (core->anal, core->offset, atoi (input + 1));
		} else if (input[1] == '-') {
			r_anal_hint_unset_opcode (core->anal, core->offset);
		}
		break;
#endif
	case 'p': // "ahp"
		if (input[1] == ' ') {
			r_anal_hint_set_pointer (core->anal, core->offset, r_num_math (core->num, input + 1));
		} else if (input[1] == '-') { // "ahp-"
			r_anal_hint_unset_pointer (core->anal, core->offset);
		}
		break;
	case '*': // "ah*"
		if (input[1] == ' ') {
			char *ptr = strdup (r_str_trim_ro (input + 2));
			r_str_word_set0 (ptr);
			ut64 addr = r_num_math (core->num, r_str_word_get0 (ptr, 0));
			r_core_anal_hint_print (core->anal, addr, '*');
		} else {
			r_core_anal_hint_list (core->anal, input[0]);
		}
		break;
	case 'j': // "ahj"
	case '\0': // "ah"
		r_core_anal_hint_list (core->anal, input[0]);
		break;
	case '-': // "ah-"
		if (input[1]) {
			if (input[1] == '*') {
				r_anal_hint_clear (core->anal);
			} else {
				char *ptr = strdup (r_str_trim_ro (input + 1));
				ut64 addr;
				int size = 1;
				int i = r_str_word_set0 (ptr);
				if (i == 2) {
					size = r_num_math (core->num, r_str_word_get0 (ptr, 1));
				}
				const char *a0 = r_str_word_get0 (ptr, 0);
				if (a0 && *a0) {
					addr = r_num_math (core->num, a0);
				} else {
					addr = core->offset;
				}
				r_anal_hint_del (core->anal, addr, size);
				free (ptr);
			}
		} else {
			r_anal_hint_clear (core->anal);
		}
		break;
	}
}

static void agraph_print_node_dot(RANode *n, void *user) {
	char *label = strdup (n->body);
	//label = r_str_replace (label, "\n", "\\l", 1);
	if (!label || !*label) {
		r_cons_printf ("\"%s\" [URL=\"%s\", color=\"lightgray\", label=\"%s\"]\n",
				n->title, n->title, n->title);
	} else {
		r_cons_printf ("\"%s\" [URL=\"%s\", color=\"lightgray\", label=\"%s\\n%s\"]\n",
				n->title, n->title, n->title, label);
	}
	free (label);
}

static void agraph_print_node(RANode *n, void *user) {
	char *encbody, *cmd;
	int len = strlen (n->body);

	if (n->body[len - 1] == '\n') {
		len--;
	}
	encbody = r_base64_encode_dyn (n->body, len);
	cmd = r_str_newf ("agn \"%s\" base64:%s\n", n->title, encbody);
	r_cons_printf (cmd);
	free (cmd);
	free (encbody);
}

static void agraph_print_edge_dot(RANode *from, RANode *to, void *user) {
	r_cons_printf ("\"%s\" -> \"%s\"\n", from->title, to->title);
}

static void agraph_print_edge(RANode *from, RANode *to, void *user) {
	r_cons_printf ("age \"%s\" \"%s\"\n", from->title, to->title);
}

static void cmd_agraph_node(RCore *core, const char *input) {
	switch (*input) {
	case ' ': { // "agn"
		char *newbody = NULL;
		char **args, *body;
		int n_args, B_LEN = strlen ("base64:");
		input++;
		args = r_str_argv (input, &n_args);
		if (n_args < 1 || n_args > 2) {
			r_cons_printf ("Wrong arguments\n");
			r_str_argv_free (args);
			break;
		}
		// strdup cause there is double free in r_str_argv_free due to a realloc call
		if (n_args > 1) {
			body = strdup (args[1]);
			if (strncmp (body, "base64:", B_LEN) == 0) {
				body = r_str_replace (body, "\\n", "", true);
				newbody = (char *)r_base64_decode_dyn (body + B_LEN, -1);
				free (body);
				if (!newbody) {
					eprintf ("Cannot allocate buffer\n");
					r_str_argv_free (args);
					break;
				}
				body = newbody;
			}
			body = r_str_append (body, "\n");
		} else {
			body = strdup ("");
		}
		r_agraph_add_node (core->graph, args[0], body);
		r_str_argv_free (args);
		free (body);
		//free newbody it's not necessary since r_str_append reallocate the space
		break;
	}
	case '-': { // "agn-"
		char **args;
		int n_args;

		input++;
		args = r_str_argv (input, &n_args);
		if (n_args != 1) {
			r_cons_printf ("Wrong arguments\n");
			r_str_argv_free (args);
			break;
		}
		r_agraph_del_node (core->graph, args[0]);
		r_str_argv_free (args);
		break;
	}
	case '?':
	default:
		r_core_cmd_help (core, help_msg_agn);
		break;
	}
}

static void cmd_agraph_edge(RCore *core, const char *input) {
	switch (*input) {
	case ' ': // "age"
	case '-': { // "age-"
		RANode *u, *v;
		char **args;
		int n_args;

		args = r_str_argv (input + 1, &n_args);
		if (n_args != 2) {
			r_cons_printf ("Wrong arguments\n");
			r_str_argv_free (args);
			break;
		}

		u = r_agraph_get_node (core->graph, args[0]);
		v = r_agraph_get_node (core->graph, args[1]);
		if (!u || !v) {
			if (!u) {
				r_cons_printf ("Node %s not found!\n", args[0]);
			} else {
				r_cons_printf ("Node %s not found!\n", args[1]);
			}
			r_str_argv_free (args);
			break;
		}
		if (*input == ' ') {
			r_agraph_add_edge (core->graph, u, v);
		} else {
			r_agraph_del_edge (core->graph, u, v);
		}
		r_str_argv_free (args);
		break;
	}
	case '?':
	default:
		r_core_cmd_help (core, help_msg_age);
		break;
	}
}

static void cmd_agraph_print(RCore *core, const char *input) {
	switch (*input) {
	case 'k': // "aggk"
	{
		Sdb *db = r_agraph_get_sdb (core->graph);
		char *o = sdb_querys (db, "null", 0, "*");
		r_cons_print (o);
		free (o);
		break;
	}
	case 'v': // "aggv"
	{
		const char *cmd = r_config_get (core->config, "cmd.graph");
		if (cmd && *cmd) {
			char *newCmd = strdup (cmd);
			if (newCmd) {
				newCmd = r_str_replace (newCmd, "ag $$", "aggd", 0);
				r_core_cmd0 (core, newCmd);
				free (newCmd);
			}
		} else {
			r_core_cmd0 (core, "agf");
		}
		break;
	}
	case 'i': // "aggi" - open current core->graph in interactive mode
	{
		RANode *ran = r_agraph_get_first_node (core->graph);
		if (ran) {
			r_agraph_set_title (core->graph, r_config_get (core->config, "graph.title"));
			r_agraph_set_curnode (core->graph, ran);
			core->graph->force_update_seek = true;
			core->graph->need_set_layout = true;
			core->graph->layout = r_config_get_i (core->config, "graph.layout");
			int ov = r_config_get_i (core->config, "scr.interactive");
			core->graph->need_update_dim = true;
			r_core_visual_graph (core, core->graph, NULL, true);
			r_config_set_i (core->config, "scr.interactive", ov);
			r_cons_show_cursor (true);
		} else {
			eprintf ("This graph contains no nodes\n");
		}
		break;
	}
	case 'd': // "aggd" - dot format
		r_cons_printf ("digraph code {\ngraph [bgcolor=white];\n"
			"node [color=lightgray, style=filled shape=box "
			"fontname=\"Courier\" fontsize=\"8\"];\n");
		r_agraph_foreach (core->graph, agraph_print_node_dot, NULL);
		r_agraph_foreach_edge (core->graph, agraph_print_edge_dot, NULL);
		r_cons_printf ("}\n");
		break;
	case '*': // "agg*" -
		r_agraph_foreach (core->graph, agraph_print_node, NULL);
		r_agraph_foreach_edge (core->graph, agraph_print_edge, NULL);
		break;
	case '?':
		r_core_cmd_help (core, help_msg_agg);
		break;
	default:
		core->graph->can->linemode = r_config_get_i (core->config, "graph.linemode");
		core->graph->can->color = r_config_get_i (core->config, "scr.color");
		r_agraph_set_title (core->graph,
			r_config_get (core->config, "graph.title"));
		r_agraph_print (core->graph);
		break;
	}
}

static void cmd_anal_graph(RCore *core, const char *input) {
	RList *list;
	const char *arg;
	switch (input[0]) {
	case 'f': // "agf"
		switch (input[1]) {
		case 't':// "agft" - tiny graph
			r_core_visual_graph (core, NULL, NULL, 2);
			break;
		case 0:
			r_core_visual_graph (core, NULL, NULL, false);
			break;
		default:
			eprintf ("Usage: agf or agft (for tiny)\n");
			break;
		}
		break;
	case '-': // "ag-"
		r_agraph_reset (core->graph);
		break;
	case 'n': // "agn"
		cmd_agraph_node (core, input + 1);
		break;
	case 'e': // "age"
		cmd_agraph_edge (core, input + 1);
		break;
	case 'g': // "agg"
		cmd_agraph_print (core, input + 1);
		break;
	case 's': // "ags"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), 0);
		break;
	case 't': // "agt"
		list = r_core_anal_graph_to (core, r_num_math (core->num, input + 1), 0);
		if (list) {
			RListIter *iter, *iter2;
			RList *list2;
			RAnalBlock *bb;
			r_list_foreach (list, iter, list2) {
				r_list_foreach (list2, iter2, bb) {
					r_cons_printf ("-> 0x%08" PFMT64x "\n", bb->addr);
				}
			}
			r_list_purge (list);
			free (list);
		}
		break;
	case 'C': // "agC"
		r_core_anal_coderefs (core, UT64_MAX, input[1] == 'j'? 2: 1);
		break;
	case 'r': // "refs"
		switch (input[1]) {
		case '*':
		case 'j':
		case ' ':
		case 0:
			{
				ut64 addr = input[2]? r_num_math (core->num, input + 2): core->offset;
				r_core_anal_codexrefs (core, addr, '*');
			}
			break;
		default:
			eprintf ("|ERROR| Usage: agr[*j]\n");
			break;
		}
		break;
	case 'c': // "agc"
		if (input[1] == '*') {
			ut64 addr = input[2]? r_num_math (core->num, input + 2): UT64_MAX;
			r_core_anal_coderefs (core, addr, '*');
		} else if (input[1] == 'j') {
			ut64 addr = input[2]? r_num_math (core->num, input + 2): UT64_MAX;
			r_core_anal_coderefs (core, addr, 2);
		} else if (input[1] == ' ') {
			ut64 addr = input[2]? r_num_math (core->num, input + 1): UT64_MAX;
			r_core_anal_coderefs (core, addr, 1);
		} else {
			eprintf ("|ERROR| Usage: agc[j*] ([addr])\n");
		}
		break;
	case 'j': // "agj"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), R_CORE_ANAL_JSON);
		break;
	case 'J': // "agJ"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), R_CORE_ANAL_JSON | R_CORE_ANAL_JSON_FORMAT_DISASM);
		break;
	case 'k': // "agk"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), R_CORE_ANAL_KEYVALUE);
		break;
	case 'l': // "agl"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), R_CORE_ANAL_GRAPHLINES);
		break;
	case 'a': // "aga"
		r_core_anal_graph (core, r_num_math (core->num, input + 1), 0);
		break;
	case 'd': // "agd"
		r_core_anal_graph (core, r_num_math (core->num, input + 1),
				R_CORE_ANAL_GRAPHBODY | R_CORE_ANAL_GRAPHDIFF);
		break;
	case 'v': // "agv"
		if (r_config_get_i (core->config, "graph.web")) {
			r_core_cmd0 (core, "=H /graph/");
		} else {
			const char *cmd = r_config_get (core->config, "cmd.graph");
			if (cmd && *cmd) {
				r_core_cmd0 (core, cmd);
			} else {
				r_core_cmd0 (core, "agf");
			}
		}
		break;
	case '?': // "ag?"
		r_core_cmd_help (core, help_msg_ag);
		break;
	case ' ': // "ag"
		arg = strchr (input, ' ');
		r_core_anal_graph (core, r_num_math (core->num, arg? arg + 1: NULL),
				R_CORE_ANAL_GRAPHBODY);
		break;
	case 0:
		eprintf ("|ERROR| Usage: ag [addr]\n");
		break;
	default:
		eprintf ("See ag?\n");
		break;
	}
}

R_API int r_core_anal_refs(RCore *core, const char *input) {
	int cfg_debug = r_config_get_i (core->config, "cfg.debug");
	ut64 from, to;
	char *ptr;
	int rad, n;
	if (*input == '?') {
		r_core_cmd_help (core, help_msg_aar);
		return 0;
	}

	if (*input == 'j' || *input == '*') {
		rad = *input;
		input++;
	} else {
		rad = 0;
	}

	from = to = 0;
	ptr = r_str_trim_head (strdup (input));
	n = r_str_word_set0 (ptr);
	if (!n) {
		// get boundaries of current memory map, section or io map
		if (cfg_debug) {
			RDebugMap *map = r_debug_map_get (core->dbg, core->offset);
			if (map) {
				from = map->addr;
				to = map->addr_end;
			}
		} else {
			RList *list = r_core_get_boundaries_prot (core, R_IO_EXEC, NULL, "anal");
			RListIter *iter;
			RIOMap* map;
			r_list_foreach (list, iter, map) {
				from = map->itv.addr;
				to = r_itv_end (map->itv);
				if (r_cons_is_breaked ()) {
					break;
				}
				if (!from && !to) {
					eprintf ("Cannot determine xref search boundaries\n");
				} else if (to - from > UT32_MAX) {
					eprintf ("Skipping huge range\n");
				} else {
					r_core_anal_search_xrefs (core, from, to, rad);
				}
			}
			free (ptr);
			return 1;
		}
	} else if (n == 1) {
		from = core->offset;
		to = core->offset + r_num_math (core->num, r_str_word_get0 (ptr, 0));
	} else {
		eprintf ("Invalid number of arguments\n");
	}
	free (ptr);

	if (from == UT64_MAX && to == UT64_MAX) {
		return false;
	}
	if (!from && !to) {
		return false;
	}
	if (to - from > r_io_size (core->io)) {
		return false;
	}
	return r_core_anal_search_xrefs (core, from, to, rad);
}

static const char *oldstr = NULL;

static void rowlog(RCore *core, const char *str) {
	int use_color = core->print->flags & R_PRINT_FLAGS_COLOR;
	bool verbose = r_config_get_i (core->config, "scr.prompt");
	oldstr = str;
	if (!verbose) {
		return;
	}
	if (use_color) {
		eprintf ("[ ] "Color_YELLOW"%s\r["Color_RESET, str);
	} else {
		eprintf ("[ ] %s\r[", str);
	}
}

static void rowlog_done(RCore *core) {
	int use_color = core->print->flags & R_PRINT_FLAGS_COLOR;
	bool verbose = r_config_get_i (core->config, "scr.prompt");
	if (verbose) {
		if (use_color)
			eprintf ("\r"Color_GREEN"[x]"Color_RESET" %s\n", oldstr);
		else eprintf ("\r[x] %s\n", oldstr);
	}
}

static int compute_coverage(RCore *core) {
	RListIter *iter;
	SdbListIter *iter2;
	RAnalFunction *fcn;
	RIOSection *sec;
	int cov = 0;
	r_list_foreach (core->anal->fcns, iter, fcn) {
		ls_foreach (core->io->sections, iter2, sec) {
			if (sec->flags & 1) {
				ut64 section_end = sec->vaddr + sec->vsize;
				ut64 s = r_anal_fcn_realsize (fcn);
				if (fcn->addr >= sec->vaddr && (fcn->addr + s) < section_end) {
					cov += s;
				}
			}
		}
	}
	return cov;
}

static int compute_code (RCore* core) {
	int code = 0;
	SdbListIter *iter;
	RIOSection *sec;
	ls_foreach (core->io->sections, iter, sec) {
		if (sec->flags & 1) {
			code += sec->vsize;
		}
	}
	return code;
}

static int compute_calls(RCore *core) {
	RListIter *iter;
	RAnalFunction *fcn;
	RList *xrefs;
	int cov = 0;
	r_list_foreach (core->anal->fcns, iter, fcn) {
		xrefs = r_anal_fcn_get_xrefs (core->anal, fcn);
		if (xrefs) {
			cov += r_list_length (xrefs);
			r_list_free (xrefs);
			xrefs = NULL;
		}
	}
	return cov;
}

static void r_core_anal_info (RCore *core, const char *input) {
	int fcns = r_list_length (core->anal->fcns);
	int strs = r_flag_count (core->flags, "str.*");
	int syms = r_flag_count (core->flags, "sym.*");
	int imps = r_flag_count (core->flags, "sym.imp.*");
	int code = compute_code (core);
	int covr = compute_coverage (core);
	int call = compute_calls (core);
	int xrfs = r_anal_xrefs_count (core->anal);
	int cvpc = (code > 0)? (covr * 100 / code): 0;
	if (*input == 'j') {
		r_cons_printf ("{\"fcns\":%d", fcns);
		r_cons_printf (",\"xrefs\":%d", xrfs);
		r_cons_printf (",\"calls\":%d", call);
		r_cons_printf (",\"strings\":%d", strs);
		r_cons_printf (",\"symbols\":%d", syms);
		r_cons_printf (",\"imports\":%d", imps);
		r_cons_printf (",\"covrage\":%d", covr);
		r_cons_printf (",\"codesz\":%d", code);
		r_cons_printf (",\"percent\":%d}\n", cvpc);
	} else {
		r_cons_printf ("fcns    %d\n", fcns);
		r_cons_printf ("xrefs   %d\n", xrfs);
		r_cons_printf ("calls   %d\n", call);
		r_cons_printf ("strings %d\n", strs);
		r_cons_printf ("symbols %d\n", syms);
		r_cons_printf ("imports %d\n", imps);
		r_cons_printf ("covrage %d\n", covr);
		r_cons_printf ("codesz  %d\n", code);
		r_cons_printf ("percent %d%%\n", cvpc);
	}
}

static void cmd_anal_aad(RCore *core, const char *input) {
	RListIter *iter;
	RAnalRef *ref;
	RList *list = r_list_newf (NULL);
	r_anal_xrefs_from (core->anal, list, "xref", R_ANAL_REF_TYPE_DATA, UT64_MAX);
	r_list_foreach (list, iter, ref) {
		if (r_io_is_valid_offset (core->io, ref->addr, false)) {
			r_core_anal_fcn (core, ref->at, ref->addr, R_ANAL_REF_TYPE_NULL, 1);
		}
	}
	r_list_free (list);
}


static bool archIsArmOrThumb(RCore *core) {
	RAsm *as = core ? core->assembler : NULL;
	if (as && as->cur && as->cur->arch) {
		if (r_str_startswith (as->cur->arch, "mips")) {
			return true;
		}
		if (r_str_startswith (as->cur->arch, "arm")) {
			if (as->bits < 64) {
				return true;
			}
		}
	}
	return false;
}

const bool archIsMips (RCore *core) {
	return strstr (core->assembler->cur->name, "mips");
}

void _CbInRangeAav(RCore *core, ut64 from, ut64 to, int vsize, bool asterisk, int count) {
	bool isarm = archIsArmOrThumb (core);
	if (isarm) {
		if (to & 1) {
			// .dword 0x000080b9 in reality is 0x000080b8
			to--;
			r_anal_hint_set_bits (core->anal, to, 16);
			// can we assume is gonna be always a function?
		} else {
			r_core_seek_archbits (core, from);
			ut64 bits = r_config_get_i (core->config, "asm.bits");
			r_anal_hint_set_bits (core->anal, from, bits);
		}
	} else {
		bool ismips = archIsMips (core);
		if (ismips) {
			if (from % 4 || to % 4) {
				eprintf ("False positive\n");
				return;
			}
		}
	}
	if (asterisk) {
		r_cons_printf ("ax 0x%"PFMT64x " 0x%"PFMT64x "\n", to, from);
		r_cons_printf ("Cd %d @ 0x%"PFMT64x "\n", vsize, from);
		r_cons_printf ("f+ aav.0x%08"PFMT64x "= 0x%08"PFMT64x, to, to);
	} else {
#if 1
		r_anal_ref_add (core->anal, to, from, ' ');
		r_meta_add (core->anal, 'd', from, from + vsize, NULL);
		if (!r_flag_get_at (core->flags, to, false)) {
			char *name = r_str_newf ("aav.0x%08"PFMT64x, to);
			r_flag_set (core->flags, name, to, vsize);
			free (name);
		}
#else
		r_core_cmdf (core, "ax 0x%"PFMT64x " 0x%"PFMT64x, to, from);
		r_core_cmdf (core, "Cd %d @ 0x%"PFMT64x, vsize, from);
		r_core_cmdf (core, "f+ aav.0x%08"PFMT64x "= 0x%08"PFMT64x, to, to);
#endif
	}
}

static void cmd_anal_aav(RCore *core, const char *input) {
#define seti(x,y) r_config_set_i(core->config, x, y);
#define geti(x) r_config_get_i(core->config, x);
	ut64 o_align = geti ("search.align");
	bool asterisk = strchr (input, '*');;
	bool is_debug = r_config_get_i (core->config, "cfg.debug");

	// pre
	int archAlign = r_anal_archinfo (core->anal, R_ANAL_ARCHINFO_ALIGN);
	seti ("search.align", archAlign);

	int vsize = 4; // 32bit dword
	if (core->assembler->bits == 64) {
		vsize = 8;
	}

	// body
	r_cons_break_push (NULL, NULL);
	if (is_debug) {
		RList *list = r_core_get_boundaries_prot (core, 0, "dbg.map", "anal");
		RListIter *iter;
		RIOMap *map;
		r_list_foreach (list, iter, map) {
			if (r_cons_is_breaked ()) {
				break;
			}
			eprintf ("aav: from 0x%"PFMT64x" to 0x%"PFMT64x"\n", map->itv.addr, r_itv_end (map->itv));
			(void)r_core_search_value_in_range (core, map->itv,
				map->itv.addr, r_itv_end (map->itv), vsize, asterisk, _CbInRangeAav);
		}
		r_list_free (list);
	} else {
		RList *list = r_core_get_boundaries_prot (core, 0, NULL, "anal");
		RListIter *iter, *iter2;
		RIOMap *map, *map2;
		ut64 from = UT64_MAX;
		ut64 to = UT64_MAX;
		// find values pointing to non-executable regions
		r_list_foreach (list, iter2, map2) {
			if (r_cons_is_breaked ()) {
				break;
			}
			//TODO: Reduce multiple hits for same addr
			from = r_itv_begin (map2->itv);
			to = r_itv_end (map2->itv);
			eprintf ("Value from 0x%08"PFMT64x " to 0x%08" PFMT64x "\n", from, to);
			r_list_foreach (list, iter, map) {
				ut64 begin = map->itv.addr;
				ut64 end = r_itv_end (map->itv);
				if (r_cons_is_breaked ()) {
					break;
				}
				if (end - begin > UT32_MAX) {
					eprintf ("Skipping huge range\n");
					continue;
				}
				eprintf ("aav: 0x%08"PFMT64x"-0x%08"PFMT64x" in 0x%"PFMT64x"-0x%"PFMT64x"\n", from, to, begin, end);
				(void)r_core_search_value_in_range (core, map->itv, from, to, vsize, asterisk, _CbInRangeAav);
				}
		}
		r_list_free (list);
	}
	r_cons_break_pop ();
	// end
	seti ("search.align", o_align);
}

static bool should_aav(RCore *core) {
	// Don't aav on x86 for now
	if (r_str_startswith (r_config_get (core->config, "asm.arch"), "x86")) {
		return false;
	}
	return true;
}

static int cmd_anal_all(RCore *core, const char *input) {
	switch (*input) {
	case '?': r_core_cmd_help (core, help_msg_aa); break;
	case 'b': // "aab"
		cmd_anal_blocks (core, input + 1);
		break; // "aab"
	case 'f': // "aaf"
		{
			int analHasnext = r_config_get_i (core->config, "anal.hasnext");
			r_config_set_i (core->config, "anal.hasnext", true);
			r_core_cmd0 (core, "afr@@c:isq");
			r_config_set_i (core->config, "anal.hasnext", analHasnext);
		}
		break;
	case 'c': // "aac"
		switch (input[1]) {
		case '*':
			cmd_anal_calls (core, input + 1, true); break; // "aac*"
		default:
			cmd_anal_calls (core, input + 1, false); break; // "aac"
		}
	case 'j': cmd_anal_jumps (core, input + 1); break; // "aaj"
	case '*': // "aa*"
		r_core_cmd0 (core, "af @@ sym.*");
		r_core_cmd0 (core, "af @@ entry*");
		break;
	case 'd': // "aad"
		cmd_anal_aad (core, input);
		break;
	case 'v': // "aav"
		cmd_anal_aav (core, input);
		break;
	case 'u': // "aau" - print areas not covered by functions
		r_core_anal_nofunclist (core, input + 1);
		break;
	case 'i': // "aai"
		r_core_anal_info (core, input + 1);
		break;
	case 's': // "aas"
		r_core_cmd0 (core, "af @@= `isq~[0]`");
		r_core_cmd0 (core, "af @@ entry*");
		break;
	case 'n': // "aan"
		r_core_anal_autoname_all_fcns (core);
		break; //aan
	case 'p': // "aap"
		if (*input == '?') {
			// TODO: accept parameters for ranges
			eprintf ("Usage: /aap   ; find in memory for function preludes");
		} else {
			r_core_search_preludes (core);
		}
		break;
	case '\0': // "aa"
	case 'a':
		if (input[0] && (input[1] == '?' || (input[1] && input[2] == '?'))) {
			r_cons_println ("Usage: See aa? for more help");
		} else {
			char *dh_orig = NULL;
			if (!strncmp (input, "aaaaa", 5)) {
				eprintf ("An r2 developer is coming to your place to manually analyze this program. Please wait for it\n");
				if (r_config_get_i (core->config, "scr.interactive")) {
					r_cons_any_key (NULL);
				}
				goto jacuzzi;
			}
			ut64 curseek = core->offset;
			rowlog (core, "Analyze all flags starting with sym. and entry0 (aa)");
			r_cons_break_push (NULL, NULL);
			r_cons_break_timeout (r_config_get_i (core->config, "anal.timeout"));
			r_core_anal_all (core);
			rowlog_done (core);
			dh_orig = core->dbg->h
					? strdup (core->dbg->h->name)
					: strdup ("esil");
			if (core->io && core->io->desc && core->io->desc->plugin && !core->io->desc->plugin->isdbg) {
				//use dh_origin if we are debugging
				R_FREE (dh_orig);
			}
			if (r_cons_is_breaked ()) {
				goto jacuzzi;
			}
			r_cons_clear_line (1);
			if (*input == 'a') { // "aaa"
				if (dh_orig && strcmp (dh_orig, "esil")) {
					r_core_cmd0 (core, "dL esil");
				}
				int c = r_config_get_i (core->config, "anal.calls");
				if (should_aav (core)) {
					rowlog (core, "\nAnalyze value pointers (aav)");
					r_core_cmd0 (core, "aav");
					if (r_cons_is_breaked ()) {
						goto jacuzzi;
					}
				}
				r_config_set_i (core->config, "anal.calls", 1);
				r_core_cmd0 (core, "s $S");
				rowlog (core, "Analyze len bytes of instructions for references (aar)");
				if (r_cons_is_breaked ()) {
					goto jacuzzi;
				}
				(void)r_core_anal_refs (core, ""); // "aar"
				rowlog_done (core);
				if (r_cons_is_breaked ()) {
					goto jacuzzi;
				}
				rowlog (core, "Analyze function calls (aac)");
				(void) cmd_anal_calls (core, "", false); // "aac"
				r_core_seek (core, curseek, 1);
				// rowlog (core, "Analyze data refs as code (LEA)");
				// (void) cmd_anal_aad (core, NULL); // "aad"
				rowlog_done (core);
				if (r_cons_is_breaked ()) {
					goto jacuzzi;
				}
				if (input[1] == 'a') { // "aaaa"
					bool ioCache = r_config_get_i (core->config, "io.pcache");
					r_config_set_i (core->config, "io.pcache", 1);
					rowlog (core, "Emulate code to find computed references (aae)");
					r_core_cmd0 (core, "aae $SS @ $S");
					rowlog_done (core);
					rowlog (core, "Analyze consecutive function (aat)");
					r_core_cmd0 (core, "aat");
					rowlog_done (core);
					// drop cache writes is no cache was
					if (!ioCache) {
						r_core_cmd0 (core, "wc-*");
					}
					r_config_set_i (core->config, "io.pcache", ioCache);
				} else {
					rowlog (core, "Use -AA or aaaa to perform additional experimental analysis.");
					rowlog_done (core);
				}
				r_config_set_i (core->config, "anal.calls", c);
				rowlog (core, "Constructing a function name for fcn.* and sym.func.* functions (aan)");
				if (r_cons_is_breaked ()) {
					goto jacuzzi;
				}
				if (r_config_get_i (core->config, "anal.autoname")) {
					r_core_anal_autoname_all_fcns (core);
					rowlog_done (core);
				}
				if (input[1] == 'a') { // "aaaa"
					bool ioCache = r_config_get_i (core->config, "io.pcache");
					r_config_set_i (core->config, "io.pcache", 1);
					if (sdb_count (core->anal->sdb_zigns) > 0) {
						rowlog (core, "Check for zignature from zigns folder (z/)");
						r_core_cmd0 (core, "z/");
					}
					rowlog (core, "Type matching analysis for all functions (afta)");
					r_core_cmd0 (core, "afta");
					rowlog_done (core);
					if (!ioCache) {
						r_core_cmd0 (core, "wc-*");
					}
					r_config_set_i (core->config, "io.pcache", ioCache);
				}
				r_core_cmd0 (core, "s-");
				if (dh_orig) {
					r_core_cmdf (core, "dL %s;dpa", dh_orig);
				}
			}
			r_core_seek (core, curseek, 1);
		jacuzzi:
			flag_every_function (core);
			r_cons_break_pop ();
			R_FREE (dh_orig);
		}
		break;
	case 't': { // "aat"
		ut64 cur = core->offset;
		bool hasnext = r_config_get_i (core->config, "anal.hasnext");
		RListIter *iter;
		RIOMap* map;
		RList *list = r_core_get_boundaries_prot (core, R_IO_EXEC, NULL, "anal");
		r_list_foreach (list, iter, map) {
			r_core_seek (core, map->itv.addr, 1);
			r_config_set_i (core->config, "anal.hasnext", 1);
			r_core_cmd0 (core, "afr");
			r_config_set_i (core->config, "anal.hasnext", hasnext);
		}
		r_core_seek (core, cur, 1);
		break;
	}
	case 'T': // "aaT"
		cmd_anal_aftertraps (core, input + 1);
		break;
	case 'E': // "aaE"
		r_core_cmd0 (core, "aef @@f");
		break;
	case 'e': // "aae"
		if (input[1]) {
			const char *len = (char *) input + 1;
			char *addr = strchr (input + 2, ' ');
			if (addr) {
				*addr++ = 0;
			}
			r_core_anal_esil (core, len, addr);
		} else {
			ut64 at = core->offset;
			RIOMap* map;
			RListIter *iter;
			RList *list = r_core_get_boundaries_prot (core, -1, NULL, "anal");
			r_list_foreach (list, iter, map) {
				r_core_seek (core, map->itv.addr, 1);
				r_core_anal_esil (core, "$SS", NULL);
			}
			r_core_seek (core, at, 1);
		}
		break;
	case 'r':
		(void)r_core_anal_refs (core, input + 1);
		break;
	default:
		r_core_cmd_help (core, help_msg_aa);
		break;
	}

	return true;
}

static bool anal_fcn_data (RCore *core, const char *input) {
	RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, -1);
	ut32 fcn_size = r_anal_fcn_size (fcn);
	if (fcn) {
		int i;
		bool gap = false;
		ut64 gap_addr = UT64_MAX;
		char *bitmap = calloc (1, fcn_size);
		if (bitmap) {
			RAnalBlock *b;
			RListIter *iter;
			r_list_foreach (fcn->bbs, iter, b) {
				int f = b->addr - fcn->addr;
				int t = R_MIN (f + b->size, fcn_size);
				if (f >= 0) {
					while (f < t) {
						bitmap[f++] = 1;
					}
				}
			}
		}
		for (i = 0; i < fcn_size; i++) {
			ut64 here = fcn->addr + i;
			if (bitmap && bitmap[i]) {
				if (gap) {
					r_cons_printf ("Cd %d @ 0x%08"PFMT64x"\n", here - gap_addr, gap_addr);
					gap = false;
				}
				gap_addr = UT64_MAX;
			} else {
				if (!gap) {
					gap = true;
					gap_addr = here;
				}
			}
		}
		if (gap) {
			r_cons_printf ("Cd %d @ 0x%08"PFMT64x"\n", fcn->addr + fcn_size - gap_addr, gap_addr);
			gap = false;
		}
		free (bitmap);
		return true;
	}
	return false;
}

static int cmpaddr (const void *_a, const void *_b) {
	const RAnalFunction *a = _a, *b = _b;
	return a->addr - b->addr;
}

static bool anal_fcn_data_gaps (RCore *core, const char *input) {
	ut64 end = UT64_MAX;
	RAnalFunction *fcn;
	RListIter *iter;
	int i, wordsize = (core->assembler->bits == 64)? 8: 4;
	r_list_sort (core->anal->fcns, cmpaddr);
	r_list_foreach (core->anal->fcns, iter, fcn) {
		if (end != UT64_MAX) {
			int range = fcn->addr - end;
			if (range > 0) {
				for (i = 0; i + wordsize < range; i+= wordsize) {
					r_cons_printf ("Cd %d @ 0x%08"PFMT64x"\n", wordsize, end + i);
				}
				r_cons_printf ("Cd %d @ 0x%08"PFMT64x"\n", range - i, end + i);
				//r_cons_printf ("Cd %d @ 0x%08"PFMT64x"\n", range, end);
			}
		}
		end = fcn->addr + r_anal_fcn_size (fcn);
	}
	return true;
}

static void cmd_anal_rtti(RCore *core, const char *input) {
	switch (input[0]) {
	case '\0': // "avr"
	case 'j': // "avrj"
		r_anal_rtti_print_at_vtable (core->anal, core->offset, input[0]);
		break;
	case 'a': // "avra"
		r_anal_rtti_print_all (core->anal, input[1]);
		break;
	default :
		r_core_cmd_help (core, help_msg_av);
		break;
	}
}

static void cmd_anal_virtual_functions(RCore *core, const char* input) {
	switch (input[0]) {
	case '\0': // "av"
	case '*': // "av*"
	case 'j': // "avj"
		r_anal_list_vtables (core->anal, input[0]);
		break;
	case 'r': // "avr"
		cmd_anal_rtti (core, input + 1);
		break;
	default :
		r_core_cmd_help (core, help_msg_av);
		break;
	}
}

static int cmd_anal(void *data, const char *input) {
	const char *r;
	RCore *core = (RCore *)data;
	ut32 tbs = core->blocksize;

	switch (input[0]) {
	case 'p': // "ap"
		{
			const ut8 *prelude = (const ut8*)"\xe9\x2d"; //:fffff000";
			const int prelude_sz = 2;
			const int bufsz = 4096;
			ut8 *buf = calloc (1, bufsz);
			ut64 off = core->offset;
			if (input[1] == ' ') {
				off = r_num_math (core->num, input+1);
				r_io_read_at (core->io, off - bufsz + prelude_sz, buf, bufsz);
			} else {
				r_io_read_at (core->io, off - bufsz + prelude_sz, buf, bufsz);
			}
			//const char *prelude = "\x2d\xe9\xf0\x47"; //:fffff000";
			r_mem_reverse (buf, bufsz);
			//r_print_hexdump (NULL, off, buf, bufsz, 16, -16);
			const ut8 *pos = r_mem_mem (buf, bufsz, prelude, prelude_sz);
			if (pos) {
				int delta = (size_t)(pos - buf);
				eprintf ("POS = %d\n", delta);
				eprintf ("HIT = 0x%"PFMT64x"\n", off - delta);
				r_cons_printf ("0x%08"PFMT64x"\n", off - delta);
			} else {
				eprintf ("Cannot find prelude\n");
			}
			free (buf);
		}
		break;
	case '8':
		{
			ut8 *buf = malloc (strlen (input) + 1);
			if (buf) {
				int len = r_hex_str2bin (input + 1, buf);
				if (len > 0) {
					core_anal_bytes (core, buf, len, 0, input[1]);
				}
				free (buf);
			}
		}
		break;
	case 'b':
		if (input[1] == 'b') { // "abb"
			core_anal_bbs (core, input + 2);
		} else if (input[1] == 'r') { // "abr"
			core_anal_bbs_range (core, input + 2);
		} else if (input[1] == ' ' || !input[1]) {
			// find block
			ut64 addr = core->offset;
			if (input[1]) {
				addr = r_num_math (core->num, input + 1);
			}
			r_core_cmdf (core, "afbi @ 0x%"PFMT64x, addr);
		} else {
			r_core_cmd_help (core, help_msg_ab);
		}
		break;
	case 'i': cmd_anal_info (core, input + 1); break; // "ai"
	case 'r': cmd_anal_reg (core, input + 1); break;  // "ar"
	case 'e': cmd_anal_esil (core, input + 1); break; // "ae"
	case 'o': cmd_anal_opcode (core, input + 1); break; // "ao"
	case 'O': cmd_anal_bytes (core, input + 1); break; // "aO"
	case 'F': // "aF"
		r_core_anal_fcn (core, core->offset, UT64_MAX, R_ANAL_REF_TYPE_NULL, 1);
		break;
	case 'f': // "af"
		{
		int res = cmd_anal_fcn (core, input);
		if (!res) {
			return false;
		}
		}
		break;
	case 'n': // 'an'
		{
		const char *name = NULL;
		bool use_json = false;

		if (input[1] == 'j') {
			use_json = true;
			input++;
		}

		if (input[1] == ' ') {
			name = input + 1;
			while (name[0] == ' ') {
				name++;
			}
			char *end = strchr (name, ' ');
			if (end) {
				*end = '\0';
			}
			if (*name == '\0') {
				name = NULL;
			}
		}

		cmd_an (core, use_json, name);
		}
		break;
	case 'g': // "ag"
		cmd_anal_graph (core, input + 1);
		break;
	case 's': // "as"
		cmd_anal_syscall (core, input + 1);
		break;
	case 'v': // "av"
		cmd_anal_virtual_functions (core, input + 1);
		break;
	case 'x': // "ax"
		if (!cmd_anal_refs (core, input + 1)) {
			return false;
		}
		break;
	case 'a': // "aa"
		if (!cmd_anal_all (core, input + 1))
			return false;
		break;
	case 'c': // "ac"
		{
		RList *hooks;
		RListIter *iter;
		RAnalCycleHook *hook;
		char *instr_tmp = NULL;
		int ccl = input[1]? r_num_math (core->num, &input[2]): 0; //get cycles to look for
		int cr = r_config_get_i (core->config, "asm.cmt.right");
		int fun = r_config_get_i (core->config, "asm.functions");
		int li = r_config_get_i (core->config, "asm.lines");
		int xr = r_config_get_i (core->config, "asm.xrefs");

		r_config_set_i (core->config, "asm.cmt.right", true);
		r_config_set_i (core->config, "asm.functions", false);
		r_config_set_i (core->config, "asm.lines", false);
		r_config_set_i (core->config, "asm.xrefs", false);

		hooks = r_core_anal_cycles (core, ccl); //analyse
		r_cons_clear_line (1);
		r_list_foreach (hooks, iter, hook) {
			instr_tmp = r_core_disassemble_instr (core, hook->addr, 1);
			r_cons_printf ("After %4i cycles:\t%s", (ccl - hook->cycles), instr_tmp);
			r_cons_flush ();
			free (instr_tmp);
		}
		r_list_free (hooks);

		r_config_set_i (core->config, "asm.cmt.right", cr); //reset settings
		r_config_set_i (core->config, "asm.functions", fun);
		r_config_set_i (core->config, "asm.lines", li);
		r_config_set_i (core->config, "asm.xrefs", xr);
		}
		break;
	case 'd': // "ad"
		switch (input[1]) {
		case 'f': // "adf"
			if (input[2] == 'g') {
				anal_fcn_data_gaps (core, input + 1);
			} else {
				anal_fcn_data (core, input + 1);
			}
			break;
		case 't': // "adt"
			cmd_anal_trampoline (core, input + 2);
			break;
		case ' ': { // "ad"
			const int default_depth = 1;
			const char *p;
			int a, b;
			a = r_num_math (core->num, input + 2);
			p = strchr (input + 2, ' ');
			b = p? r_num_math (core->num, p + 1): default_depth;
			if (a < 1) {
				a = 1;
			}
			if (b < 1) {
				b = 1;
			}
			r_core_anal_data (core, core->offset, a, b, 0);
		} break;
		case 'k': // "adk"
			r = r_anal_data_kind (core->anal,
					core->offset, core->block, core->blocksize);
			r_cons_println (r);
			break;
		case '\0': // "ad"
			r_core_anal_data (core, core->offset, 2 + (core->blocksize / 4), 1, 0);
			break;
		case '4': // "ad4"
			r_core_anal_data (core, core->offset, 2 + (core->blocksize / 4), 1, 4);
			break;
		case '8': // "ad8"
			r_core_anal_data (core, core->offset, 2 + (core->blocksize / 4), 1, 8);
			break;
		default:
			r_core_cmd_help (core, help_msg_ad);
			break;
		}
		break;
	case 'h': // "ah"
		cmd_anal_hint (core, input + 1);
		break;
	case '!': // "a!"
		if (core->anal && core->anal->cur && core->anal->cur->cmd_ext) {
			return core->anal->cur->cmd_ext (core->anal, input + 1);
		} else {
			r_cons_printf ("No plugins for this analysis plugin\n");
		}
		break;
	default:
		r_core_cmd_help (core, help_msg_a);
#if 0
		r_cons_printf ("Examples:\n"
			" f ts @ `S*~text:0[3]`; f t @ section..text\n"
			" f ds @ `S*~data:0[3]`; f d @ section..data\n"
			" .ad t t+ts @ d:ds\n",
			NULL);
#endif
		break;
	}
	if (tbs != core->blocksize) {
		r_core_block_size (core, tbs);
	}
	if (r_cons_is_breaked ()) {
		r_cons_clear_line (1);
	}
	return 0;
}
