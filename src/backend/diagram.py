"""
Code to generate and save the Mermaid diagram of langgraph

"""


from pathlib import Path


OUT = Path("graph_out")
OUT.mkdir(exist_ok=True)


def write_graph_png(graph_view, output_path="graph.png"):
    "Write a LangGraph view PNG and return its path."
    output_path = Path(output_path).with_suffix(".png")
    output_path.write_bytes(graph_view.draw_mermaid_png())
    return output_path


def show(graph, output_path="graph.png", render_inline=False):
    "Render a compiled LangGraph to a PNG file."
    png_path = write_graph_png(graph.get_graph(), output_path)
    print(f"PNG diagram written to: {png_path.resolve()}")

    if render_inline:
        from IPython.display import Image, display

        display(Image(filename=str(png_path)))

    return png_path


def save_graph_png(app, name="graph"):
    "Save a compiled LangGraph PNG under graph_out."
    return show(app, output_path=OUT / f"{name}.png")


def show_graph(graph_or_name, name="graph"):
    "Display a saved PNG by name, or save/display a compiled LangGraph passed by main."
    if hasattr(graph_or_name, "get_graph"):
        show(graph_or_name, output_path=OUT / f"{name}.png", render_inline=True)
        return

    from IPython.display import Image, Markdown, display

    png_path = OUT / f"{graph_or_name}.png"
    if png_path.exists():
        display(Image(filename=str(png_path)))
    else:
        display(Markdown(f"PNG not on disk: `graph_out/{graph_or_name}.png`"))
