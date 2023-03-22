# Mercury's Contributing guide

## Where to start?

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

If you're new at contributing to open-source projects, please, check the [Issues secion](https://github.com/BBVA/mercury-dataschema/issues). There you'll find ideas and already reported bugs you can start messing around with. As a rule of thumb, they'll be flagged with the 'good-first-issue' tag.

When you start working on an issue, it’s a good idea to assign the issue to yourself, so nobody else duplicates the work on it. GitHub restricts assigning issues to maintainers of the project only.

If for whatever reason you are not able to continue working with the issue, please try to unassign it, so other people know it’s available again. You can check the list of assigned issues, since people may not be working in them anymore. If you want to work on one that is assigned, feel free to kindly ask the current assignee if you can take it (please allow at least a week of inactivity before considering work in the issue discontinued).

Mercury has a dedicated team which will review your contributions and help you on your contribution journey. However, you can also join as a reviewer for other contributions and help polishing them.

## Bug reports and enhacement ideas

You can report bugs and/or request features via the [Issues secion](https://github.com/BBVA/mercury-dataschema/issues). 

Please, keep in mind that posts there must contain as much information as possible so others can reproduce your error or have a clear picture of your proposal. It's often a good idea to paste code snippets to enrich your examples.


## Working with the codebase

The code is hosted on GitHub. To contribute you will need to sign up for a free GitHub account. We use Git for version control to allow many people to work together on the project.

Some great resources for learning Git:

- The [Git documentation](https://git-scm.com/doc)
- The [GitHub help pages](https://help.github.com/)
- The [GitHub setup guide](https://help.github.com/set-up-git-redirect)

Once you feel confortable with Git you can start by **forking** the repository. This will make a complete copy of the codebase under your profile, which you'll be using from now on. Clone it on your machine.

### Creating a feature branch

On your local repository – the cloned fork you made – you may start by creating a feature branch on which you'll implement your feature. As a naming convention we use the following:

- `feature/MY_FEATURE_NAME`: For any new feature named `MY_FEATURE_NAME`.
- `fix/MY_FIX_NAME`: For any fix named `FIX_NAME`.
- `doc/MY_IMPROVEMENT`: For any improvement to the documentation (README, docstrings, tutorials...).

You can create and move to a new branch with:

```
git checkout -b feature/my_branch_name
```

When creating this branch, make sure your main branch is up to date with the latest upstream main version. To update your local main branch, you can do:

```
git checkout main
git pull upstream main --ff-only
```

### Testing your new feature

In case you're including a new feature or modifying the existing code, it's expected you implement new tests accordingly. We don't ask for any specific coverage percentage, but tipically it should be high for the classes/functions you're testing. When designing your tests, please, focus on testing all the possible input/output conditions rather than on raw coverage percentage. 

### Committing your changes

Once you have implemented your feature (and added tests for it, if necessary), you can make a commit with the changes on your local copy of the repository.

```
git add FILE_OR_FILES_YOU_MODIFIED
git commit 
```

Please, add a helpful descriptions of the changes so they can be easily trackable in the future.

### Pushing your changes to GitHub

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```
git push origin feature/my_branch_name
```

Now your code is on GitHub, but it is not yet a part of the Mercury project. For that to happen, a pull request needs to be submitted on GitHub.

### Make a Pull Request (PR)

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be looked at and eventually merged into the main version. This pull request and its associated changes will eventually be committed to the main branch and available in the next release. To submit a pull request:

1. Navigate to your repository on GitHub.
2. Click on the "Compare & pull request" button.
3. You can then click on "Commits and Files Changed" to make sure everything looks okay one last time.
4. Write a descriptive title and a detailed description in case any third person read your PR in the future.
5. Click "Send Pull Request".

This request then goes to the repository maintainers, and they will review the code.

## Help and support 

This library is currently maintained by a dedicated team of data scientists and machine learning engineers from BBVA AI Factory. 

### Documentation
website: https://bbva.github.io/mercury-dataschema/

### Email 
mercury.group@bbva.com
