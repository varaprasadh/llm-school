import { Component } from "react";

/**
 * Contains render errors so a single broken visualization or chapter shows a
 * friendly inline message instead of taking down the whole page. Used around
 * each Figure's children and around the lazy chapter body.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Surface during development; harmless in production.
    if (import.meta.env?.DEV) {
      // eslint-disable-next-line no-console
      console.error("ErrorBoundary caught:", error, info);
    }
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div className="my-4 rounded-xl border border-rose-500/40 bg-rose-500/5 p-4 text-sm text-rose-200">
            <div className="font-semibold">⚠️ This {this.props.label || "component"} hit a snag.</div>
            <div className="mt-1 text-rose-200/70">
              The rest of the page is fine — try reloading. ({String(this.state.error?.message || this.state.error)})
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
