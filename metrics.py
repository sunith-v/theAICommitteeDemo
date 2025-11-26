import json
import os
import csv
from datetime import datetime
from typing import List, Dict, Any

class ComprehensiveMetrics:
    """
    A comprehensive metrics tracker for multi-agent, LLM-based data processing pipelines.

    This class tracks overall performance, cost, and token usage, as well as
    detailed per-row and per-agent statistics to provide deep insights into the
    pipeline's operational characteristics.
    """

    def __init__(self, price_input: float, price_output: float, price_cached: float = None):
        self.price_per_input_token = price_input
        self.price_per_output_token = price_output
        self.price_per_cached_token = price_cached if price_cached is not None else price_input

        self.total_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0

        self._row_metrics: Dict[str, Dict[str, Any]] = {}
        self._row_start_times: Dict[str, datetime] = {}

        self._agent_call_counts: Dict[str, int] = {}
        self._agent_token_usage: Dict[str, Dict[str, int]] = {}
        self._agent_cost_usd: Dict[str, float] = {}
        self._agent_latency_total_s: Dict[str, float] = {}
        self._agent_latency_counts: Dict[str, int] = {}
        self._agent_retry_counts: Dict[str, int] = {}

        self._agent_active_start: Dict[str, datetime] = {}

        self.discovery_attempts = 0
        self.discovery_found_points = 0
        self.discovery_total_latency_s = 0.0
        self.discovery_total_cost_usd = 0.0
        self.discovery_accepts = 0

        # Web scraping metrics
        self.web_scraping_requests = 0
        self.web_scraping_cache_hits = 0
        self.web_scraping_successes = 0
        self.web_scraping_failures = 0
        self.web_scraping_content_quality_scores = []

        # Remediation effectiveness metrics
        self.remediation_attempts = 0
        self.remediation_successes = 0
        self.remediation_quality_improvements = []
        self.remediation_types_used = {}

        # Error pattern tracking
        self.error_patterns = {}
        self.common_failure_modes = {}

    def start_row_processing(self, row_id: str):
        self._row_start_times[row_id] = datetime.now()
        self._row_metrics[row_id] = {
            'row_id': row_id,
            'status': 'PROCESSING',
            'api_calls': 0,
            'prompt_tokens': 0,
            'cached_tokens': 0,
            'completion_tokens': 0,
            'cost': 0.0,
            'latency': 0.0,
            'error': None,
            'start_time': self._row_start_times[row_id].isoformat()
        }

    def end_row_processing(self, row_id: str, status: str, reason: str = None):
        if row_id in self._row_start_times:
            end_time = datetime.now()
            latency = (end_time - self._row_start_times[row_id]).total_seconds()
            
            self._row_metrics[row_id].update({
                'status': status,
                'latency': latency,
                'reason': reason,
                'end_time': end_time.isoformat()
            })
            if row_id in self._row_start_times:
                del self._row_start_times[row_id]

    def update_request_metrics(self, response: Any, agent_name: str, row_id: str):
        self.total_requests += 1

        token_usage = getattr(response, 'response_metadata', {}).get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens_details = token_usage.get("prompt_tokens_details", {})
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_cached_tokens += cached_tokens
        self.total_completion_tokens += completion_tokens

        # Calculate cost: cached tokens at cached price, non-cached at input price
        non_cached_prompt_tokens = prompt_tokens - cached_tokens
        request_cost = (cached_tokens * self.price_per_cached_token) + \
                       (non_cached_prompt_tokens * self.price_per_input_token) + \
                       (completion_tokens * self.price_per_output_token)

        if row_id in self._row_metrics:
            self._row_metrics[row_id]['api_calls'] += 1
            self._row_metrics[row_id]['prompt_tokens'] += prompt_tokens
            self._row_metrics[row_id]['cached_tokens'] += cached_tokens
            self._row_metrics[row_id]['completion_tokens'] += completion_tokens
            self._row_metrics[row_id]['cost'] += request_cost

        self._agent_call_counts[agent_name] = self._agent_call_counts.get(agent_name, 0) + 1
        if agent_name not in self._agent_token_usage:
            self._agent_token_usage[agent_name] = {'prompt': 0, 'cached': 0, 'completion': 0}
        self._agent_token_usage[agent_name]['prompt'] += prompt_tokens
        self._agent_token_usage[agent_name]['cached'] += cached_tokens
        self._agent_token_usage[agent_name]['completion'] += completion_tokens
        self._agent_cost_usd[agent_name] = self._agent_cost_usd.get(agent_name, 0.0) + request_cost
        
    def update(self, response: Any, agent_name: str = None, row_id: str = None):
        try:
            if agent_name is not None and row_id is not None:
                self.update_request_metrics(response, agent_name, row_id)
                return
        except Exception:
            pass

        self.total_requests += 1
        token_usage = getattr(response, 'response_metadata', {}).get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens_details = token_usage.get("prompt_tokens_details", {})
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_cached_tokens += cached_tokens
        self.total_completion_tokens += completion_tokens
        
    def log_failure(self, row_id: str = None, agent_name: str = None):
        self.failed_requests += 1
        if agent_name is not None:
            self._agent_call_counts[agent_name] = self._agent_call_counts.get(agent_name, 0) + 1

    def start_agent(self, agent_name: str):
        self._agent_active_start[agent_name] = datetime.now()

    def end_agent(self, agent_name: str):
        start = self._agent_active_start.get(agent_name)
        if start:
            delta = (datetime.now() - start).total_seconds()
            self._agent_latency_total_s[agent_name] = self._agent_latency_total_s.get(agent_name, 0.0) + delta
            self._agent_latency_counts[agent_name] = self._agent_latency_counts.get(agent_name, 0) + 1
            del self._agent_active_start[agent_name]

    def record_retry(self, agent_name: str):
        self._agent_retry_counts[agent_name] = self._agent_retry_counts.get(agent_name, 0) + 1

    def record_discovery_attempt(self):
        self.discovery_attempts += 1

    def record_discovery_result(self, found_count: int, duration_s: float, cost_usd: float):
        self.discovery_found_points += max(0, int(found_count))
        self.discovery_total_latency_s += max(0.0, duration_s)
        self.discovery_total_cost_usd += max(0.0, cost_usd)

    def record_discovery_accepts(self, accepts: int):
        self.discovery_accepts += max(0, int(accepts))

    def increment_success(self):
        self.total_requests += 1
        
    def increment_failure(self):
        self.failed_requests += 1
        self.total_requests += 1

    def track_web_scraping(self, url: str, success: bool, cached: bool = False, content_quality: float = None):
        """Track web scraping metrics"""
        self.web_scraping_requests += 1
        if cached:
            self.web_scraping_cache_hits += 1
        if success:
            self.web_scraping_successes += 1
        else:
            self.web_scraping_failures += 1
        if content_quality is not None:
            self.web_scraping_content_quality_scores.append(content_quality)

    def track_remediation(self, success: bool, quality_improvement: float = None, remediation_type: str = None):
        """Track remediation effectiveness"""
        self.remediation_attempts += 1
        if success:
            self.remediation_successes += 1
        if quality_improvement is not None:
            self.remediation_quality_improvements.append(quality_improvement)
        if remediation_type:
            self.remediation_types_used[remediation_type] = self.remediation_types_used.get(remediation_type, 0) + 1

    def track_error_pattern(self, error_type: str, context: str = None):
        """Track error patterns and common failure modes"""
        self.common_failure_modes[error_type] = self.common_failure_modes.get(error_type, 0) + 1
        if context:
            error_key = f"{error_type}:{context}"
            self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1

    def calculate_summary(self) -> Dict[str, Any]:
        all_rows = list(self._row_metrics.values())
        processed_rows = [r for r in all_rows if r['status'] != 'PROCESSING']

        # Always include basic structure even with no rows
        if not processed_rows:
            summary = {
                "overall_summary": {"total_rows_processed": 0, "successful_rows": 0, "rejected_rows": 0, "discovered_new_rows": 0, "skipped_rows (errors)": 0, "success_rate": 0.0},
                "performance": {"total_processing_time (s)": 0.0, "avg_latency_per_row (s)": 0.0, "throughput (rows/sec)": 0.0},
                "cost_and_api_usage": {"total_cost ($)": 0.0, "avg_cost_per_row ($)": 0.0, "cost_per_successful_row ($)": 0.0, "total_api_requests": self.total_requests, "failed_api_requests": self.failed_requests, "total_prompt_tokens": self.total_prompt_tokens, "total_cached_tokens": self.total_cached_tokens, "total_completion_tokens": self.total_completion_tokens, "total_tokens": self.total_prompt_tokens + self.total_completion_tokens, "per_agent_cost ($)": {}},
                "agent_breakdown": {},
                "discovery_roi": {"attempts": self.discovery_attempts, "found_points": self.discovery_found_points, "total_latency (s)": self.discovery_total_latency_s, "total_cost ($)": self.discovery_total_cost_usd, "accepts_after_discovery": self.discovery_accepts, "roi_accept_rate": (self.discovery_accepts / self.discovery_found_points) if self.discovery_found_points else 0.0}
            }
            # Add the new metrics sections even with no processed rows
            summary["web_scraping"] = {
                "total_requests": self.web_scraping_requests,
                "cache_hits": self.web_scraping_cache_hits,
                "cache_hit_rate": (self.web_scraping_cache_hits / self.web_scraping_requests) if self.web_scraping_requests else 0.0,
                "successes": self.web_scraping_successes,
                "failures": self.web_scraping_failures,
                "success_rate": (self.web_scraping_successes / self.web_scraping_requests) if self.web_scraping_requests else 0.0,
                "avg_content_quality_score": sum(self.web_scraping_content_quality_scores) / len(self.web_scraping_content_quality_scores) if self.web_scraping_content_quality_scores else 0.0
            }
            summary["remediation_effectiveness"] = {
                "attempts": self.remediation_attempts,
                "successes": self.remediation_successes,
                "success_rate": (self.remediation_successes / self.remediation_attempts) if self.remediation_attempts else 0.0,
                "avg_quality_improvement": sum(self.remediation_quality_improvements) / len(self.remediation_quality_improvements) if self.remediation_quality_improvements else 0.0,
                "remediation_types": dict(sorted(self.remediation_types_used.items(), key=lambda x: x[1], reverse=True))
            }
            summary["error_analysis"] = {
                "common_failure_modes": dict(sorted(self.common_failure_modes.items(), key=lambda x: x[1], reverse=True)),
                "error_patterns": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True))
            }
            return summary
        else:
            successful_rows = [r for r in processed_rows if r['status'].startswith('ACCEPT')]
        
        total_processed = len(processed_rows)
        total_successful = len(successful_rows)
        
        latencies = [r['latency'] for r in processed_rows]
        total_cost = sum(r['cost'] for r in processed_rows)

        agent_avg_latency = {
            agent: (self._agent_latency_total_s.get(agent, 0.0) / self._agent_latency_counts.get(agent, 1))
            for agent in self._agent_latency_total_s.keys() | self._agent_latency_counts.keys()
        }

        summary = {
            "overall_summary": {
                "total_rows_processed": total_processed,
                "successful_rows": total_successful,
                "rejected_rows": len([r for r in processed_rows if r['status'] == 'REJECT']),
                "discovered_new_rows": len([r for r in processed_rows if r['status'] == 'DISCOVERED_NEW']),
                "skipped_rows (errors)": len([r for r in processed_rows if r['status'].startswith('SKIP')]),
                "success_rate": total_successful / total_processed if total_processed else 0,
            },
            "performance": {
                "total_processing_time (s)": sum(latencies),
                "avg_latency_per_row (s)": sum(latencies) / total_processed if total_processed else 0,
                "throughput (rows/sec)": total_processed / sum(latencies) if sum(latencies) > 0 else 0,
            },
            "cost_and_api_usage": {
                "total_cost ($)": total_cost,
                "avg_cost_per_row ($)": total_cost / total_processed if total_processed else 0,
                "cost_per_successful_row ($)": total_cost / total_successful if total_successful else 0,
                "total_api_requests": self.total_requests,
                "failed_api_requests": self.failed_requests,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "per_agent_cost ($)": dict(sorted(self._agent_cost_usd.items())),
            },
            "agent_breakdown": {
                agent: {
                    "calls": self._agent_call_counts.get(agent, 0),
                    "prompt_tokens": self._agent_token_usage.get(agent, {}).get('prompt', 0),
                    "cached_tokens": self._agent_token_usage.get(agent, {}).get('cached', 0),
                    "completion_tokens": self._agent_token_usage.get(agent, {}).get('completion', 0),
                    "avg_latency (s)": agent_avg_latency.get(agent, 0.0),
                    "retries": self._agent_retry_counts.get(agent, 0),
                } for agent in sorted(self._agent_call_counts.keys())
            }
        }
        summary["discovery_roi"] = {
            "attempts": self.discovery_attempts,
            "found_points": self.discovery_found_points,
            "total_latency (s)": self.discovery_total_latency_s,
            "total_cost ($)": self.discovery_total_cost_usd,
            "accepts_after_discovery": self.discovery_accepts,
            "roi_accept_rate": (self.discovery_accepts / self.discovery_found_points) if self.discovery_found_points else 0.0
        }
        summary["web_scraping"] = {
            "total_requests": self.web_scraping_requests,
            "cache_hits": self.web_scraping_cache_hits,
            "cache_hit_rate": (self.web_scraping_cache_hits / self.web_scraping_requests) if self.web_scraping_requests else 0.0,
            "successes": self.web_scraping_successes,
            "failures": self.web_scraping_failures,
            "success_rate": (self.web_scraping_successes / self.web_scraping_requests) if self.web_scraping_requests else 0.0,
            "avg_content_quality_score": sum(self.web_scraping_content_quality_scores) / len(self.web_scraping_content_quality_scores) if self.web_scraping_content_quality_scores else 0.0
        }
        summary["remediation_effectiveness"] = {
            "attempts": self.remediation_attempts,
            "successes": self.remediation_successes,
            "success_rate": (self.remediation_successes / self.remediation_attempts) if self.remediation_attempts else 0.0,
            "avg_quality_improvement": sum(self.remediation_quality_improvements) / len(self.remediation_quality_improvements) if self.remediation_quality_improvements else 0.0,
            "remediation_types": dict(sorted(self.remediation_types_used.items(), key=lambda x: x[1], reverse=True))
        }
        summary["error_analysis"] = {
            "common_failure_modes": dict(sorted(self.common_failure_modes.items(), key=lambda x: x[1], reverse=True)),
            "error_patterns": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True))
        }
        return summary

    def report(self):
        summary = self.calculate_summary()
        if not summary:
            print("No metrics collected.")
            return

        print("\n" + "="*80)
        print("COMPREHENSIVE METRICS REPORT")
        print("="*80)
        
        os_summary = summary['overall_summary']
        print("\n[ Processing Summary ]")
        for key, value in os_summary.items():
            val_str = f"{value:.2%}" if "rate" in key else f"{value}"
            print(f"  {key.replace('_', ' ').title():<28}: {val_str}")
            
        perf_summary = summary['performance']
        print("\n[ Performance ]")
        for key, value in perf_summary.items():
            print(f"  {key.replace('_', ' ').title():<28}: {value:.4f}")

        cost_summary = summary['cost_and_api_usage']
        print("\n[ Cost & API Usage ]")
        for key, value in cost_summary.items():
            if isinstance(value, dict):
                print(f"  {key.replace('_', ' ').title():<28}:")
                for sub_k, sub_v in value.items():
                    try:
                        sub_val_str = f"${sub_v:.6f}" if ("$" in key or "$" in sub_k) else f"{sub_v:,}"
                    except (TypeError, ValueError):
                        sub_val_str = str(sub_v)
                    print(f"    - {sub_k}: {sub_val_str}")
                continue

            try:
                val_str = f"${value:.6f}" if "$" in key else f"{value:,}"
            except (TypeError, ValueError):
                val_str = str(value)
            print(f"  {key.replace('_', ' ').title():<28}: {val_str}")

        agent_summary = summary['agent_breakdown']
        print("\n[ Agent Breakdown ]")
        print(f"  {'Agent Name':<28} | {'Calls':>8} | {'Prompt Tokens':>15} | {'Completion Tokens':>18}")
        print("  " + "-"*75)
        for agent, data in agent_summary.items():
            print(f"  {agent:<28} | {data['calls']:>8,} | {data['prompt_tokens']:>15,} | {data['completion_tokens']:>18,}")

        print("="*80)

    def save_json(self, output_path: str, extras: Dict[str, Any] = None):
        summary = self.calculate_summary()
        if extras:
            summary['experiment_details'] = extras
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)
            print(f"\nMetrics summary saved to: {output_path}")
        except IOError as e:
            print(f"Error writing metrics to {output_path}: {e}")

    def save_csv(self, output_path: str, extras: Dict[str, Any] = None):
        """
        Save a flattened, run-level metrics summary to a single-row CSV.

        This is designed for experiment tracking: one row per run with key
        performance, cost, and token-usage statistics plus any extra metadata.
        """
        summary = self.calculate_summary()
        if not summary:
            print("No metrics collected, skipping CSV export.")
            return

        # Base experiment details
        row: Dict[str, Any] = {}
        if extras:
            # Shallow copy so callers can reuse their dict
            row.update(extras)

        overall = summary.get("overall_summary", {})
        perf = summary.get("performance", {})
        cost = summary.get("cost_and_api_usage", {})
        discovery = summary.get("discovery_roi", {})
        web = summary.get("web_scraping", {})
        remediation = summary.get("remediation_effectiveness", {})

        # Overall
        row.update({
            "total_rows_processed": overall.get("total_rows_processed", 0),
            "successful_rows": overall.get("successful_rows", 0),
            "rejected_rows": overall.get("rejected_rows", 0),
            "discovered_new_rows": overall.get("discovered_new_rows", 0),
            "skipped_rows_errors": overall.get("skipped_rows (errors)", 0),
            "success_rate": overall.get("success_rate", 0.0),
        })

        # Performance
        row.update({
            "total_processing_time_s": perf.get("total_processing_time (s)", 0.0),
            "avg_latency_per_row_s": perf.get("avg_latency_per_row (s)", 0.0),
            "throughput_rows_per_s": perf.get("throughput (rows/sec)", 0.0),
        })

        # Cost & API
        row.update({
            "total_cost_usd": cost.get("total_cost ($)", 0.0),
            "avg_cost_per_row_usd": cost.get("avg_cost_per_row ($)", 0.0),
            "cost_per_successful_row_usd": cost.get("cost_per_successful_row ($)", 0.0),
            "total_api_requests": cost.get("total_api_requests", 0),
            "failed_api_requests": cost.get("failed_api_requests", 0),
            "total_prompt_tokens": cost.get("total_prompt_tokens", 0),
            "total_cached_tokens": cost.get("total_cached_tokens", 0),
            "total_completion_tokens": cost.get("total_completion_tokens", 0),
            "total_tokens": cost.get("total_tokens", 0),
        })

        # Discovery ROI
        row.update({
            "discovery_attempts": discovery.get("attempts", 0),
            "discovery_found_points": discovery.get("found_points", 0),
            "discovery_total_latency_s": discovery.get("total_latency (s)", 0.0),
            "discovery_total_cost_usd": discovery.get("total_cost ($)", 0.0),
            "discovery_accepts_after_discovery": discovery.get("accepts_after_discovery", 0),
            "discovery_roi_accept_rate": discovery.get("roi_accept_rate", 0.0),
        })

        # Web scraping
        row.update({
            "web_requests": web.get("total_requests", 0),
            "web_cache_hits": web.get("cache_hits", 0),
            "web_cache_hit_rate": web.get("cache_hit_rate", 0.0),
            "web_successes": web.get("successes", 0),
            "web_failures": web.get("failures", 0),
            "web_success_rate": web.get("success_rate", 0.0),
            "web_avg_content_quality": web.get("avg_content_quality_score", 0.0),
        })

        # Remediation
        row.update({
            "remediation_attempts": remediation.get("attempts", 0),
            "remediation_successes": remediation.get("successes", 0),
            "remediation_success_rate": remediation.get("success_rate", 0.0),
            "remediation_avg_quality_improvement": remediation.get("avg_quality_improvement", 0.0),
        })

        # Per-agent cost can be large; store as JSON blob in a single column for convenience
        per_agent_cost = cost.get("per_agent_cost ($)", {}) or {}
        row["per_agent_cost_json"] = json.dumps(per_agent_cost, sort_keys=True)

        # Make sure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        write_header = not os.path.exists(output_path)
        try:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"\nMetrics CSV row appended to: {output_path}")
        except IOError as e:
            print(f"Error writing metrics CSV to {output_path}: {e}")