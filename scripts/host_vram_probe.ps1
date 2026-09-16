# host_vram_probe.ps1 — per-process VRAM attribution from the Windows side.

Why: inside a WSL guest, `nvidia-smi --query-compute-apps` frequently reports a
single app with `[N/A]` memory, and the Windows driver (WDDM) does not surface
per-process usage there. The Windows performance counters do:

  \GPU Process Memory(<instance>)\Local Usage    -> dedicated VRAM per process
  \GPU Process Memory(<instance>)\Shared Usage   -> system-RAM spill (sysmem fallback)

Two facts that keep this from being a mistake:

  * A WSL guest's CUDA context belongs to `vmwp.exe` (the VM process), NOT to
    the python process you see inside the guest. High vmwp.exe usage IS your
    training run. It is not a thief.
  * `Shared Usage` in the 1.5-2GB band on the training process = sysmem
    fallback is ACTIVE (the card overflowed into RAM over PCIe). That is the
    masked-OOM signature; nvidia-smi will not tell you per-process.

Cross-check before concluding anything: list compute apps INSIDE the guest.
If it shows only the trainer, there is no second consumer and high VRAM is
just the current step's real footprint.

Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File host_vram_probe.ps1
        powershell ... -File host_vram_probe.ps1 -Top 15 -SharedWarnMib 1024

Do NOT inline this logic in `python -c` or `powershell -Command`: the shell
eats `$_`, `*` and quotes and you get interpolated garbage. Keep it a .ps1.

#>
param(
    [int]$Top = 10,
    [int]$SharedWarnMib = 1024
)

$ErrorActionPreference = 'SilentlyContinue'

# Instance names look like: pid_1234_luid_0x0000_0x0000_phys_0_seg_0_node_0
$samples = (Get-Counter -Counter '\GPU Process Memory(*)\*' -SampleInterval 1 `
             -MaxSamples 1).CounterSamples

# NOTE: $pid is a reserved read-only automatic variable in PowerShell — never
# assign to it (the assignment fails silently and the whole probe returns 0
# rows, which reads as "no GPU processes"). Use $procId.
$rows = @{}
foreach ($s in $samples) {
    if ($s.InstanceName -notmatch '^pid_(\d+)_') { continue }
    $procId = [int]$Matches[1]
    if (-not $rows.ContainsKey($procId)) {
        $rows[$procId] = [pscustomobject]@{
            Pid = $procId; Name = ''; LocalMib = 0.0; SharedMib = 0.0 }
    }
    $v = [math]::Round($s.CookedValue / 1MB, 1)
    if     ($s.Path -match 'Local Usage')  { $rows[$procId].LocalMib  = $v }
    elseif ($s.Path -match 'Shared Usage') { $rows[$procId].SharedMib = $v }
}

foreach ($r in $rows.Values) {
    $p = Get-Process -Id $r.Pid
    if ($p) { $r.Name = $p.ProcessName } else { $r.Name = '<gone>' }
}

$out = $rows.Values | Sort-Object -Descending LocalMib | Select-Object -First $Top

"{0,-8} {1,-22} {2,10} {3,10}  {4}" -f 'PID','NAME','LOCAL(MiB)','SHARED(MiB)','NOTE'
"-" * 78
foreach ($r in $out) {
    $note = ''
    if ($r.Name -eq 'vmwp')  { $note = 'WSL VM (guest CUDA context lives here)' }
    if ($r.SharedMib -gt $SharedWarnMib) {
        $note = if ($note) { "$note; SYSMEM SPILL" } else { 'SYSMEM SPILL' }
    }
    "{0,-8} {1,-22} {2,10} {3,10}  {4}" -f $r.Pid, $r.Name, $r.LocalMib, $r.SharedMib, $note
}

""
"Total dedicated held by listed processes: {0} MiB" -f `
   [math]::Round(($out | Measure-Object -Property LocalMib -Sum).Sum, 0)
"If a training process shows SharedMib >> 0, the card overflowed into system RAM:"
"  compare tok/s against the run's own history before calling it a slowdown,"
"  and read power.draw (a card at 60W/170W with 100% util is waiting on PCIe, not computing)."
