#!/usr/bin/env python3
"""Interactive test runner for the negotiation framework.

This script provides a comprehensive testing interface for all negotiation
components, allowing selective testing, visual output, and performance analysis.
"""

import sys
import os
import time
import argparse
import unittest
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add the tests to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test modules
import test_negotiation_components
import test_negotiation_integration
import test_negotiation_scenarios


class NegotiationTestRunner:
    """Main test runner for the negotiation framework."""
    
    def __init__(self, verbose: bool = True, save_results: bool = False):
        """Initialize the test runner.
        
        Args:
            verbose: Whether to show detailed output
            save_results: Whether to save test results to file
        """
        self.verbose = verbose
        self.save_results = save_results
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def print_subheader(self, title: str):
        """Print a formatted subheader."""
        print("\n" + "-"*60)
        print(f"  {title}")
        print("-"*60)
    
    def run_component_tests(self) -> unittest.TestResult:
        """Run unit tests for individual components."""
        self.print_subheader("Running Component Unit Tests")
        result = test_negotiation_components.run_component_tests(self.verbose)
        self.results['components'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful()
        }
        return result
    
    def run_integration_tests(self) -> unittest.TestResult:
        """Run integration tests for component interactions."""
        self.print_subheader("Running Integration Tests")
        result = test_negotiation_integration.run_integration_tests(self.verbose)
        self.results['integration'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful()
        }
        return result
    
    def run_scenario_tests(self) -> unittest.TestResult:
        """Run complete scenario tests."""
        self.print_subheader("Running Scenario Tests")
        result = test_negotiation_scenarios.run_scenario_tests(self.verbose)
        self.results['scenarios'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful()
        }
        return result
    
    def run_all_tests(self):
        """Run all test categories."""
        self.print_header("NEGOTIATION FRAMEWORK COMPREHENSIVE TEST SUITE")
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # Run each test category
        component_result = self.run_component_tests()
        integration_result = self.run_integration_tests()
        scenario_result = self.run_scenario_tests()
        
        self.end_time = time.time()
        
        # Print overall summary
        self.print_test_summary()
        
        # Save results if requested
        if self.save_results:
            self.save_test_results()
        
        # Return overall success
        return (component_result.wasSuccessful() and 
                integration_result.wasSuccessful() and 
                scenario_result.wasSuccessful())
    
    def run_selected_tests(self, categories: List[str]):
        """Run only selected test categories.
        
        Args:
            categories: List of test categories to run
                       ('components', 'integration', 'scenarios')
        """
        self.print_header("NEGOTIATION FRAMEWORK SELECTIVE TEST SUITE")
        print(f"Running categories: {', '.join(categories)}")
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        all_successful = True
        
        if 'components' in categories:
            result = self.run_component_tests()
            all_successful = all_successful and result.wasSuccessful()
        
        if 'integration' in categories:
            result = self.run_integration_tests()
            all_successful = all_successful and result.wasSuccessful()
        
        if 'scenarios' in categories:
            result = self.run_scenario_tests()
            all_successful = all_successful and result.wasSuccessful()
        
        self.end_time = time.time()
        
        # Print summary
        self.print_test_summary()
        
        # Save results if requested
        if self.save_results:
            self.save_test_results()
        
        return all_successful
    
    def print_test_summary(self):
        """Print a comprehensive test summary."""
        self.print_header("TEST EXECUTION SUMMARY")
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        # Component breakdown
        print("\n📊 Test Results by Category:")
        print("-" * 50)
        
        for category, stats in self.results.items():
            status = "✅ PASS" if stats['success'] else "❌ FAIL"
            print(f"\n{category.capitalize()} Tests: {status}")
            print(f"  • Tests run: {stats['tests_run']}")
            print(f"  • Failures: {stats['failures']}")
            print(f"  • Errors: {stats['errors']}")
            
            total_tests += stats['tests_run']
            total_failures += stats['failures']
            total_errors += stats['errors']
        
        # Overall statistics
        print("\n" + "="*50)
        print("📈 Overall Statistics:")
        print(f"  • Total tests run: {total_tests}")
        print(f"  • Total failures: {total_failures}")
        print(f"  • Total errors: {total_errors}")
        print(f"  • Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
        
        if self.end_time and self.start_time:
            duration = self.end_time - self.start_time
            print(f"  • Execution time: {duration:.2f} seconds")
        
        # Final status
        print("\n" + "="*50)
        if total_failures == 0 and total_errors == 0:
            print("🎉 ALL TESTS PASSED! The negotiation framework is working correctly.")
        else:
            print("⚠️  Some tests failed. Please review the output above for details.")
    
    def save_test_results(self):
        """Save test results to an HTML file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"negotiation_test_results_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Negotiation Framework Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .stats {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .category {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Negotiation Framework Test Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <div class="stats">
        <table>
            <tr>
                <th>Category</th>
                <th>Tests Run</th>
                <th>Failures</th>
                <th>Errors</th>
                <th>Status</th>
            </tr>
"""
        
        for category, stats in self.results.items():
            status_class = 'pass' if stats['success'] else 'fail'
            status_text = 'PASS' if stats['success'] else 'FAIL'
            html_content += f"""
            <tr>
                <td>{category.capitalize()}</td>
                <td>{stats['tests_run']}</td>
                <td>{stats['failures']}</td>
                <td>{stats['errors']}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <h2>Test Categories</h2>
    
    <div class="category">
        <h3>Component Tests</h3>
        <p>Unit tests for individual negotiation components including strategies, 
           memory, cultural adaptation, theory of mind, and more.</p>
    </div>
    
    <div class="category">
        <h3>Integration Tests</h3>
        <p>Tests for component interactions and the integration framework that 
           coordinates multiple modules.</p>
    </div>
    
    <div class="category">
        <h3>Scenario Tests</h3>
        <p>End-to-end tests of complete negotiation scenarios including bilateral, 
           multi-party, cross-cultural, and information asymmetry scenarios.</p>
    </div>
    
</body>
</html>
"""
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"\n📄 Test results saved to: {filename}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for the Concordia negotiation framework"
    )
    
    parser.add_argument(
        '--category',
        choices=['all', 'components', 'integration', 'scenarios'],
        nargs='+',
        default=['all'],
        help='Test categories to run'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed test output'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save test results to HTML file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a quick smoke test of key functionality'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = NegotiationTestRunner(
        verbose=args.verbose,
        save_results=args.save
    )
    
    # Determine what to run
    if args.quick:
        print("🚀 Running quick smoke test...")
        # Run a subset of critical tests
        categories = ['components']
    elif 'all' in args.category:
        print("🔍 Running all tests...")
        success = runner.run_all_tests()
    else:
        categories = args.category
        success = runner.run_selected_tests(categories)
    
    # Exit with appropriate code
    if 'all' not in args.category:
        success = runner.run_selected_tests(categories)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()