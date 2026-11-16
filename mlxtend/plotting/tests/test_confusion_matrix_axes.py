import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.plotting import plot_confusion_matrix


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("provide_figure", [False, True])
@pytest.mark.parametrize("colorbar", [False, True])
@pytest.mark.parametrize("show_normed", [False, True])
def test_supplied_axes_owns_labels_ticks_image_and_colorbar(
    provide_figure, colorbar, show_normed
):
    figure, (target, other) = plt.subplots(1, 2)
    other.set_xlabel("unrelated x")
    other.set_ylabel("unrelated y")
    other.set_xticks([0.2, 0.8], labels=["left", "right"])
    other.set_yticks([0.1, 0.9], labels=["bottom", "top"])
    plt.sca(other)
    conf_mat = np.array([[2, 1], [0, 3]])
    kwargs = {"figure": figure} if provide_figure else {}
    returned_figure, returned_axes = plot_confusion_matrix(
        conf_mat,
        axis=target,
        class_names=["negative", "positive"],
        colorbar=colorbar,
        show_normed=show_normed,
        **kwargs,
    )
    assert returned_figure is figure
    assert returned_axes is target
    assert target.get_xlabel() == "predicted label"
    assert target.get_ylabel() == "true label"
    assert [tick.get_text() for tick in target.get_xticklabels()] == [
        "negative",
        "positive",
    ]
    assert [tick.get_text() for tick in target.get_yticklabels()] == [
        "negative",
        "positive",
    ]
    assert other.get_xlabel() == "unrelated x"
    assert other.get_ylabel() == "unrelated y"
    assert [tick.get_text() for tick in other.get_xticklabels()] == ["left", "right"]
    assert [tick.get_text() for tick in other.get_yticklabels()] == ["bottom", "top"]
    assert not other.images
    assert len(target.images) == 1
    expected = (
        conf_mat / conf_mat.sum(axis=1, keepdims=True) if show_normed else conf_mat
    )
    assert_array_equal(target.images[0].get_array(), expected)
    assert len(figure.axes) == 2 + int(colorbar)
    assert plt.gca() is other
    if colorbar:
        assert target.images[0].colorbar.ax.figure is figure


@pytest.mark.parametrize("colorbar", [False, True])
def test_axes_only_uses_its_figure_when_another_figure_is_current(colorbar):
    target_figure, target = plt.subplots()
    other_figure, other = plt.subplots()
    returned_figure, returned_axes = plot_confusion_matrix(
        np.array([[2, 1], [0, 3]]),
        axis=target,
        colorbar=colorbar,
    )
    assert returned_figure is target_figure
    assert returned_axes is target
    assert len(target_figure.axes) == 1 + int(colorbar)
    assert len(other_figure.axes) == 1
    assert not other.images
    assert other.get_xlabel() == other.get_ylabel() == ""
    assert plt.gca() is other


@pytest.mark.parametrize("provide_figure", [False, True])
def test_default_and_figure_only_calls_keep_existing_behavior(provide_figure):
    kwargs = {"figure": plt.figure()} if provide_figure else {}
    figure, axes = plot_confusion_matrix(np.eye(2, dtype=int), colorbar=True, **kwargs)
    assert axes.figure is figure
    if provide_figure:
        assert figure is kwargs["figure"]
    assert len(figure.axes) == 2
    assert axes.get_xlabel() == "predicted label"
    assert axes.get_ylabel() == "true label"


def test_mismatched_figure_and_axes_are_rejected_before_drawing():
    figure, axes = plt.subplots()
    other_figure = plt.figure()
    with pytest.raises(ValueError, match="figure"):
        plot_confusion_matrix(np.eye(2, dtype=int), figure=other_figure, axis=axes)
    assert not axes.images
    assert not other_figure.axes
    assert axes.figure is figure


@pytest.mark.parametrize("hide_ticks", [False, True])
def test_tick_and_spine_visibility_apply_only_to_target(hide_ticks):
    figure, (target, other) = plt.subplots(1, 2)
    plt.sca(other)
    plot_confusion_matrix(
        np.eye(2, dtype=int),
        axis=target,
        hide_ticks=hide_ticks,
        hide_spines=True,
        class_names=["a", "b"],
    )
    assert all(not spine.get_visible() for spine in target.spines.values())
    assert all(spine.get_visible() for spine in other.spines.values())
    assert len(target.get_xticks()) == (0 if hide_ticks else 2)
    assert len(target.get_yticks()) == (0 if hide_ticks else 2)
