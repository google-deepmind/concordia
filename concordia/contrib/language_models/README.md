# Language Model Wrappers: A Call for Community Contribution

This document outlines the process for adding and maintaining language model
wrappers in Concordia. As the landscape of language model APIs is constantly
evolving, we rely on the community to help us keep these wrappers up-to-date and
to expand our support for a wider variety of models. Your contributions are
crucial for the continued success and utility of this project.

--------------------------------------------------------------------------------

## Why We Need Your Help

The `language_model` wrappers in Concordia provide a standardized interface for
interacting with various language models. However, the underlying APIs for these
models can change frequently, leading to breakages in our existing wrappers. To
ensure that Concordia remains a robust and reliable tool for the community, we
need your help in maintaining these wrappers and adding new ones.

By contributing to this effort, you can:

*   **Ensure continued access** to your preferred language models within
    Concordia.

*   **Expand the capabilities** of Concordia by adding support for new and
    emerging models.

*   **Become an active member** of the Concordia community and collaborate with
    other researchers and developers.

--------------------------------------------------------------------------------

## How to Contribute

We welcome and encourage pull requests that add new language model wrappers or
fix existing ones. Here’s how you can get started:

### Adding a New Language Model Wrapper

If you want to add a wrapper for a new language model, please follow these
steps:

1.  **Identify the appropriate package for your model**

    *   Each `concordia.contrib.language_models.<package>` has a corresponding
        `extras_requires` in `setup.py`.
    *   So if you will need the existing `gdm-concordia[huggingface]`
        dependencies you should add your file to
        `concordia.contrib.language_models.huggingface`.
    *   If you add a new model, create a new package, and add a new set of
        `extras_requires` in `setup.py`.

1.  **Create a new file** in the appropriate package.

    *   You can use the existing wrappers as a template.

1.  **Implement the two main functions**:

    *   `sample_text(prompt)`: This function should take a prompt string and
        return a text completion from the language model.
    *   `sample_choice(prompt, choices)`: This function should take a prompt and
        a list of choices, and return one of the choices.

1.  **Add a `_REGISTRY` entry** in
    `concordia/contrib/language_models/__init__.py` to integrate your new
    wrapper.

1.  **Prepare a pull request** with your changes.

    *   If you have added any new dependencies rebuild `requirements.txt` using
        `pip-compile setup.py examples/requirements.in`.
    *   Check that the tests pass using `./bin/test.sh`
    *   Submit your PR and we will review and merge it as soon as possible.

When implementing `sample_choice`, the approach often depends on whether you
have access to the model's internal states (i.e., weights and tokenwise
predicted log probabilities):

*   **Open-Weights Approach**: For models where you can access the tokenwise
    predicted log probabilities, the best method is to append the answer choices
    to the prompt and compare the log probabilities produced by the model
    generating each choice. This provides a more direct way of measuring the
    model's preference. The `DefaultCompletion` wrapper in `together_ai.py` is a
    good example of this approach.

*   **Closed-Weights Approach**: For models accessed via APIs that don't expose
    tokenwise predicted log probabilities (like many commercial models), the
    common approach is to ask the model in the prompt to return a response that
    exactly matches one of the provided choices. This requires careful parsing
    and error handling for cases where the model's output doesn't perfectly
    match one of the options. The `BaseGPTModel` wrapper in `base_gpt_model.py`
    demonstrates this approach.

### Maintaining Existing Wrappers

If you notice that an existing wrapper is broken due to an API change, we
encourage you to fix it. You can do this by:

1.  **Identifying the source of the breakage** in the wrapper's code.
2.  **Updating the code** to be compatible with the new API.
3.  **Submitting a pull request** with your fixes.

Your contributions will help to ensure that Concordia remains a valuable
resource for the entire community. Thank you for your support!
