from analytics.nav_spike_audit import audit_nav_spikes, save_nav_spike_audit

run_dir = "runs/sparse_switch_mv_nifty500_20260504_010810"
store_dir = "processed_data/nifty500/1m_cube_store"

result = audit_nav_spikes(
    run_dir=run_dir,
    store_dir=store_dir,
    pct_nav_change=0.20,
    top_k_symbols=20,
)

spikes_path, contrib_path = save_nav_spike_audit(result, run_dir)
print("spikes:", spikes_path)
print("contributions:", contrib_path)
print(result.spikes.head(20))
print(result.contributions.head(50))