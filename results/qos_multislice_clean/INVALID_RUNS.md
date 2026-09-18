- `multislice_slicing_sla_light_rep1`: invalid on 2026-09-16.
  Autonomous UDP probe was valid, but healthcare and industrial UDP probes had
  `udp_valid_measurement=0`, 100% packet loss, and undefined jitter. Excluded
  from all analysis and plots.
- `multislice_slicing_sla_light_rep2`: invalid.
  The Mininet topology reported "Unable to contact the remote controller at
  127.0.0.1:6633". This happened because `sudo mn -c` terminated the active
  Ryu process before the topology was launched. All TCP and UDP measurements
  were invalid and this run is excluded from all analysis.
