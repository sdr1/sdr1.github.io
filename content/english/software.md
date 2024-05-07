+++
title = "Software"
description = "Steven Rashin software"
date = "2024-04-10"
aliases = ["software"]
author = "Steven Rashin"
+++

Over the years I've developed a variety of tools to help me do research.  I've decided to collect them here.

## Detecting Common Text 

This program is used when you have thousands of different files and want to find the ones that are similar.  It works by setting a distance threshold (e.g., a cosine similarity of > 90) and then finding all the groupings of documents that meet this criteria.  

I used this program to detect mass comments in a Securities and Exchange Commission rule that weren't previously classified as being part of a mass commenting campaign.

Here's the [project repository on GitHub](https://github.com/sdr1/Detect-Mass-Comments/).  There are two versions of the program, [a basic one](https://github.com/sdr1/Detect-Mass-Comments/blob/master/detect%20mass%20comments.R) that works for most data (i.e. when your documents aren't each over 100 pages), and and [advanced version](https://github.com/sdr1/Detect-Mass-Comments/blob/master/mass_comments_advanced.R) that allows you to split comments into groups so that you can deal with large documents.

## Extracting Links from Tables in RVest 

An annoying part of webscraping is that extracting links from tables can be difficult.  Often there are links outside the table too, so you can't simply extract the links and map them directly to the table.  This script solves the problem!  [Here's the code](https://github.com/sdr1/Extract-Links-From-Table/blob/main/extract-table-links.R)

## Document Similarity 

Ask me about this!  It's the basis of my PhD. 

