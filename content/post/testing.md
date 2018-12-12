---
title: "testing"
date: 2018-02-04T12:02:23+01:00
draft: false
categories: ["DevOps", "data-engineering", "R"]
tags: ["DevOps", "data-engineering", "R"]
---

## 1. What is testing and why would you use it?

* testing or [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development) (TDD) is a discipline, which relies on writing a test for every functionality *before* creating it;

* at first the test will fail, as we have not provided the proper functionality yet. Our goal is to fullfill this functionality, so the test will pass.

In reality you modify your tests as you create the functionality or even write the tests after you are finished writing it. It's OK as long as you remember to cover all the functions with tests.

## 2. "Hello World" examples

### R (testthat)

Let's go through testing two simple functions:

```
library(testthat)

context("simple example tests")

divider <- function(a, b) {
  if (b == 0) {
    warning("One does not simply divide by zero.")
    return(NULL)
  }
  x <- a / b
  return(x)
}

test_that(
  desc = "check proper division by `divider()`",
  code = {
    expect_equal(divider(10, 5), 2)

    expect_warning(divider(2, 0))

    expect_null(suppressWarnings(divider(10, 0)))
  }
)

summer <- function(a, b) {
  x <- a + b
  if (x == 5) x <- x + 1  ## a strange bug
  return(x)
}

test_that(
  desc = "check proper summing by `summer`",
  code = {
    expect_equal(summer(2, 2), 4, info = "two and two is four")
    expect_equal(summer(2, 3), 5, info = "two and three is five")
  }
)
```

What have we done here?

* we loaded `testthat` package;

* we provided a context - this is the first message that appears in tests summary and serves as a title for this particular group of tests;

*Couriously, in order to run test properly, you *have to* provide context [link](https://stackoverflow.com/questions/50083521/error-in-xmethod-attempt-to-apply-non-function-in-testthat-test-when)*.

* we wrote a function `divider`, which divides two numbers and `summer`, which adds twwo numbers (clever!); as you can see, there is a strange bug in `summer`

* `test_that` functions belong to `testthat` package and they will check if these functions run properly;

* there are various types of `expect_...`, a pretty interesting one is `expect_fail(expect_...())`;

* you should provide additional description (`desc`) to each `test_that` function, so you could easily find which test failed; you can also provide info for every single `expect_...`.


Now we can run our tests with `testthat::test_file(<name_of_file>)` or `testthat::test_dir('tests')`, depending on where you store your functions with tests. In production, you obviously keep testing functions in separate files, preferably in a `tests` folder and each file is called `test_...`.  In that case you source all the functions you want to test simply with `source()`.



Testing our above file will result in:

```
R> test_file('test.R')                                                          
✔ | OK F W S | Context
✖ |  4 1     | simple example tests
────────────────────────────────────────────────────────────────────────────────
test.R:37: failure: check proper summing by `summer`
summer(2, 3) not equal to 5.
1/1 mismatches
[1] 6 - 5 == 1
two and three is five
────────────────────────────────────────────────────────────────────────────────

══ Results ═════════════════════════════════════════════════════════════════════
OK:       4
Failed:   1
Warnings: 0
Skipped:  0
```

Information, that 4 tests have passed, one has failed. The one that failes was in 37th line of the test file, when we were 'checking proper summing by summer'. According to `summer` two and three is not five.



## 3. Useful links

* R:

    * [Hadley Wichkam's article on testthat](https://journal.r-project.org/archive/2011/RJ-2011-002/RJ-2011-002.pdf)

    * [usethis](https://github.com/r-lib/usethis) - useful if you want to test a package

* Python:

    * [Test driven development with Python](https://learning.oreilly.com/library/view/test-driven-development-with/9781449365141/)

## 4. Subjects still to cover

* pytest, unittest, coverage - or python in general
