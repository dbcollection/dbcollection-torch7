install:
	luarocks make rocks/*

uninstall:
	luarocks remove dbcollection

test:
	th tests/test.lua