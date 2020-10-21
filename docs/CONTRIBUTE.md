# How to contribute

You are welcome to contribute to project PaddleGAN. Contributions can be:

- documentation
- source code and bug fixes
- new features or models
- solve existing issues

We sincerely appreciate your contribution. This document explains our workflow and work style.

The following guidiance tells you how to submit code.

## Workflow

1. [Fork](https://help.github.com/articles/fork-a-repo/)

    Transfer to the home page of Github [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN), and then click button `Fork` to generate the git under your own file directory, such as <https://github.com/USERNAME/PaddleGAN>.

2. Clone

    Clone remote git to local:

    ```bash
    ➜  git clone https://github.com/USERNAME/PaddleGAN
    ➜  cd PaddleGAN
    ```

3. Create local branch

    At present [Git stream branch model](http://nvie.com/posts/a-successful-git-branching-model/)  is applied to PaddleGAN to undergo task of development, test, release and maintenance.
    All development tasks of feature and bug fix should be finished in a new branch which is extended from `master` branch.

    Create and switch to a new branch with command `git checkout -b`.


    ```bash
    ➜  git checkout -b my-cool-stuff
    ```

    It is worth noting that before the checkout, you need to keep the current branch directory clean, otherwise the untracked file will be brought to the new branch, which can be viewed by  `git status` .


4. Use `pre-commit` hook

    We use the [pre-commit](http://pre-commit.com/) tool to manage Git pre-commit hooks. It helps us format the source code (C++, Python) and automatically check some basic things before committing (such as having only one EOL per file, not adding large files in Git, etc.).

    The `pre-commit` test is part of the unit test in Travis-CI. A PR that does not satisfy the hook cannot be submitted to Paddle. Install `pre-commit` first and then run it in current directory：


    ```bash
    ➜  pip install pre-commit
    ➜  pre-commit install
    ```

    Once installed, `pre-commit` checks the style of code and documentation in every commit.  We will see something like the following when you run `git commit`:

     ```
     ➜  git commit
     CRLF end-lines remover...............................(no files to check)Skipped
     yapf.................................................(no files to check)Skipped
     Check for added large files..............................................Passed
     Check for merge conflicts................................................Passed
     Check for broken symlinks................................................Passed
     Detect Private Key...................................(no files to check)Skipped
     Fix End of Files.....................................(no files to check)Skipped
     clang-formater.......................................(no files to check)Skipped
     [my-cool-stuff c703c041] add test file
      1 file changed, 0 insertions(+), 0 deletions(-)
      create mode 100644 233
     ```

    NOTE: The `yapf` installed by `pip install pre-commit` and `conda install -c conda-forge pre-commit` is slightly different. We use `pip install pre-commit`.


5. Start development

    I delete a line of README.md and create a new file in the case.

    View the current state via `git status`, which will prompt some changes to the current directory, and you can also view the file's specific changes via `git diff`.


    ```bash
    ➜  git status
    On branch test
    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)
        modified:   README.md
    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        test
    no changes added to commit (use "git add" and/or "git commit -a")
    ```

6. Test

    Add unit testing if needed.


7. Commit

    Next we cancel the modification of README.md, and submit new added test file.

    ```bash
    ➜  git checkout -- README.md
    ➜  git status
    On branch test
    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        test
    nothing added to commit but untracked files present (use "git add" to track)
    ➜  git add test
    ```

    It's required that the commit message is also given on every Git commit, through which other developers will be notified of what changes have been made. Type `git commit` to realize it.

    ```bash
    ➜  git commit
    CRLF end-lines remover...............................(no files to check)Skipped
    yapf.................................................(no files to check)Skipped
    Check for added large files..............................................Passed
    Check for merge conflicts................................................Passed
    Check for broken symlinks................................................Passed
    Detect Private Key...................................(no files to check)Skipped
    Fix End of Files.....................................(no files to check)Skipped
    clang-formater.......................................(no files to check)Skipped
    [my-cool-stuff c703c041] add test file
     1 file changed, 0 insertions(+), 0 deletions(-)
     create mode 100644 233
    ```


8. Keep pulling

    An experienced Git user pulls from the official repo often -- daily or even hourly, so they notice conflicts with others work early, and it's easier to resolve smaller conflicts.

    Check the name of current remote repository with `git remote`.

    ```bash
    ➜  git remote
    origin
    ➜  git remote -v
    origin    https://github.com/USERNAME/PaddleGAN (fetch)
    origin    https://github.com/USERNAME/PaddleGAN (push)
    ```

    origin is the name of remote repository that we clone, we create a remote host of an original PaddleGAN and name it upstream.

    ```bash
    ➜  git remote add upstream https://github.com/PaddlePaddle/PaddleGAN
    ➜  git remote
    origin
    upstream
    ```

    Get the latest code of upstream and update current branch.

    ```bash
    ➜  git fetch upstream
    ➜  git pull upstream develop
    ```

9. Push to remote repository

    Push local modification to GitHub (https://github.com/USERNAME/PaddleGAN).

    ```bash
    # submit it to remote git the branch my-cool-stuff of origin
    ➜  git push origin my-cool-stuff
    ```

    The push allows you to create a pull request, requesting owners of this [official repo](https://github.com/PaddlePaddle/PaddleGAN) to pull your change into the official one.

    To create a pull request, please follow [these steps](https://help.github.com/articles/creating-a-pull-request/).

    If your change is for fixing an issue, please write ["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/) in the description section of your pull request.  Github would close the issue when the owners merge your pull request.

    Please remember to specify some reviewers for your pull request.  If you don't know who are the right ones, please follow Github's recommendation.


10. Delete local and remote branches

    To keep your local workspace and your fork clean, you might want to remove merged branches:

    ```bash
    git push origin :my-cool-stuff
    git checkout develop
    git pull upstream develop
    git branch -d my-cool-stuff
    ```

## Code Review

- Please feel free to ping your reviewers by sending them the URL of your pull request via IM or email. Please do this after your pull request passes the CI.

- Please answer reviewers' every comment.  If you are to follow the comment, please write "Done"; please give a reason otherwise.

- If you don't want your reviewers to get overwhelmed by email notifications, you might reply their comments by [in a batch](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/).

- Reduce the unnecessary commits.  Some developers commit often.  It is recommended to append a sequence of small changes into one commit by running `git commit --amend` instead of `git commit`.


## Coding Standard

### Code Style

Our C/C++ code follows the [Google style guide](http://google.github.io/styleguide/cppguide.html).

Our Python code follows the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).

Please install pre-commit, which automatically reformat the changes to C/C++ and Python code whenever we run `git commit`.

### Unit Tests

Please remember to add related unit tests.

- For C/C++ code, please follow [`google-test` Primer](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) .
- For Python code, please use [Python's standard `unittest` package](http://pythontesting.net/framework/unittest/unittest-introduction/).
