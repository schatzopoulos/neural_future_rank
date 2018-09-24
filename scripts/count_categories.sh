#!/bin/bash
cut -f 5 "$1" | sort | uniq -c | sort -nr
